import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import HubertModel
import math



HUGE_XLARGE_PATH = "/mnt/facebook/hubert-xlarge-ls960-ft"

class RotaryEmbedding(nn.Module):
    """RoPE 旋转位置编码器"""
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 预计算 cos/sin 表（支持动态长度）
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.float32)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, dim)
        
        self.register_buffer('cos_cached', emb.cos().to(dtype), persistent=False)
        self.register_buffer('sin_cached', emb.sin().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int = None):
        """
        Args:
            x: (B, ..., seq_len, dim) - 查询/键张量
        Returns:
            x_rot: 应用 RoPE 后的张量
        """
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        
        seq_len = x.shape[-2]
        cos = self.cos_cached[:seq_len].unsqueeze(0)  # (1, seq_len, dim)
        sin = self.sin_cached[:seq_len].unsqueeze(0)
        
        # 将 cos/sin 广播到 x 的形状
        cos = cos.expand(*x.shape[:-2], -1, cos.shape[-1])
        sin = sin.expand(*x.shape[:-2], -1, sin.shape[-1])
        
        x1 = x[..., :self.dim//2, :]
        x2 = x[..., self.dim//2:, :]
        
        rotated = torch.cat((-x2, x1), dim=-1)
        x_out = (x * cos) + (rotated * sin)
        return x_out

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, rope: RotaryEmbedding):
    """将 RoPE 应用到 Q/K"""
    q_embed = rope(q, seq_len=q.shape[-2])
    k_embed = rope(k, seq_len=k.shape[-2])
    return q_embed, k_embed


class RoPETransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        # 自定义的多头注意力，以便注入 RoPE
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # RoPE 核心类（使用你提供的 RotaryEmbedding）
        self.rope = RotaryEmbedding(dim=self.head_dim)
        
        # 标准组件
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model)
        )

    def forward(self, x, src_key_padding_mask=None):
        """
        x: (S, B, D) - S 是段数
        """
        # Pre-Norm 结构
        residual = x
        x = self.norm1(x)
        
        S, B, D = x.shape
        
        # 1. 生成 Q, K, V 并切分多头
        q = self.q_proj(x).view(S, B, self.nhead, self.head_dim).transpose(0, 1) # (B, S, nhead, head_dim)
        k = self.k_proj(x).view(S, B, self.nhead, self.head_dim).transpose(0, 1)
        v = self.v_proj(x).view(S, B, self.nhead, self.head_dim).transpose(0, 1)
        
        # 2. 注入 RoPE (应用旋转)
        # 注意：RoPE 作用在 (B, nhead, S, head_dim) 的最后两维
        q = q.transpose(1, 2) # (B, nhead, S, head_dim)
        k = k.transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, self.rope)
        q = q.transpose(1, 2) # 恢复 (B, S, nhead, head_dim)
        k = k.transpose(1, 2)

        # 3. Scaled Dot-Product Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5) # (B, nhead, S, S)
        
        if src_key_padding_mask is not None:
            # 这里的 mask 需要扩展形状以匹配 attn_weights
            attn_weights = attn_weights.masked_fill(src_key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_out = torch.matmul(attn_probs, v) # (B, nhead, S, head_dim)
        
        # 4. 合并头并输出
        attn_out = attn_out.transpose(1, 2).contiguous().view(S, B, D)
        x = residual + self.dropout(self.out_proj(attn_out))
        
        # 5. MLP 层
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class EnhancedSegmentEncoder(nn.Module):
    def __init__(self, input_dim=768, segment_dim=1024, num_layers=12):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, segment_dim)
        self.duration_embed = nn.Linear(1, segment_dim)
        
        # 使用自定义的 RoPE Transformer 层
        self.layers = nn.ModuleList([
            RoPETransformerEncoderLayer(d_model=segment_dim, nhead=16) 
            for _ in range(num_layers)
        ])
        
    def forward(self, segments, durations, padding_mask):
        """
        segments: (S, B, D)
        durations: (S, B, 1) 
        padding_mask: (B, S) - True 代表是 padding
        """
        # 基础映射
        x = self.input_projection(segments)
        
        # 注入时长信息（片段内特征补偿）
        x = x + self.duration_embed(durations)
        
        # 逐层通过 RoPE Transformer
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=padding_mask)
            
        return x

    def load_xlarge_weights(self, xlarge_path: str = HUGE_XLARGE_PATH):
            print(f"Initializing Enhanced SegmentEncoder with RoPE from HuBERT XLarge...")
            
            try:
                # 1. 加载预训练模型
                hubert_xlarge = HubertModel.from_pretrained(xlarge_path, local_files_only=True)
                START_LAYER = 6
                
                # 2. 遍历你自定义的 ModuleList
                # 注意：现在的 self.layers 是 ModuleList，不再是官方的 transformer_encoder
                for i in range(len(self.layers)):
                    target_layer = self.layers[i]
                    source_layer = hubert_xlarge.encoder.layer[START_LAYER + i]
                    
                    # 获取源状态字典
                    src_sd = source_layer.state_dict()
                    # 获取目标层当前的状态字典
                    tgt_sd = target_layer.state_dict()
                    
                    # --- 手动映射开始 ---
                    # 注意力部分
                    tgt_sd['q_proj.weight'] = src_sd['attention.self.query.weight']
                    tgt_sd['q_proj.bias']   = src_sd['attention.self.query.bias']
                    tgt_sd['k_proj.weight'] = src_sd['attention.self.key.weight']
                    tgt_sd['k_proj.bias']   = src_sd['attention.self.key.bias']
                    tgt_sd['v_proj.weight'] = src_sd['attention.self.value.weight']
                    tgt_sd['v_proj.bias']   = src_sd['attention.self.value.bias']
                    tgt_sd['out_proj.weight'] = src_sd['attention.output.dense.weight']
                    tgt_sd['out_proj.bias']   = src_sd['attention.output.dense.bias']
                    
                    # LayerNorms
                    tgt_sd['norm1.weight'] = src_sd['layer_norm.weight']
                    tgt_sd['norm1.bias']   = src_sd['layer_norm.bias']
                    tgt_sd['norm2.weight'] = src_sd['final_layer_norm.weight']
                    tgt_sd['norm2.bias']   = src_sd['final_layer_norm.bias']
                    
                    # MLP 层 (Sequential)
                    tgt_sd['mlp.0.weight'] = src_sd['intermediate.dense.weight']
                    tgt_sd['mlp.0.bias']   = src_sd['intermediate.dense.bias']
                    tgt_sd['mlp.2.weight'] = src_sd['output.dense.weight']
                    tgt_sd['mlp.2.bias']   = src_sd['output.dense.bias']
                    # --- 手动映射结束 ---
                    
                    # 加载映射后的权重
                    target_layer.load_state_dict(tgt_sd)
                    
                print(f"成功将 HuBERT XLarge 权重注入 RoPE 架构（层 6-{START_LAYER + len(self.layers) - 1}）")
                
                del hubert_xlarge
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"权重注入失败: {e}")






class SegmentEncoder(nn.Module):
    # 匹配 HuBERT XLarge 的核心参数
    HUGO_XLARGE_DIM = 1024
    HUGO_XLARGE_HEADS = 16  # HuBERT XLarge 使用 16 个注意力头
    
    def __init__(self, 
                 input_dim: int = 768,        # Segment Pooling 得到的特征维度 (通常是 HuBERT Base/Large 的 768)
                 segment_dim: int = HUGO_XLARGE_DIM, # 教师模型处理的内部维度 (1024)
                 num_layers: int = 12,        # 使用 12 层 Transformer Encoder
                 num_heads: int = HUGO_XLARGE_HEADS,
                 max_segments: int = 1000,
                 dropout: float = 0.1):
        super().__init__()
        
        # 1. 输入维度投影层 (关键): 
        # 用于将 Segment Pooling 输出的特征 (e.g., 768D) 映射到 HuBERT XLarge 的内部维度 (1024D)。
        # 如果您的 HuBERT Backbone 已经是 1024D，这个层就是 nn.Identity() 或直接跳过。
        self.input_projection = nn.Linear(input_dim, segment_dim)
        
        # 2. 位置编码 (用于捕获片段的顺序信息)
        self.pos_encoder = nn.Embedding(max_segments, segment_dim) 
        self.dropout = nn.Dropout(dropout)
        
        # 3. Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=segment_dim, 
            nhead=num_heads, 
            dim_feedforward=segment_dim * 4, # 默认 MLP 扩展系数
            dropout=dropout,
            batch_first=False # (T, B, D) 格式
        )
        
        # 4. Transformer Encoder (堆叠 num_layers 个层)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)



    def forward(self, segments: torch.Tensor, padding_mask: torch.BoolTensor) -> torch.Tensor:
        """
        Args:
            segments: (T, B, D_in) - 填充后的片段序列，D_in 是输入维度 (e.g., 768)
            padding_mask: (B, T) - Transformer 的 Key Padding Mask (True=Masked)

        Returns:
            output: (T, B, D_seg) - 编码后的片段序列表示 (D_seg=1024)
        """
        time_steps, batch_size, dim_in = segments.shape
        device = segments.device
        
        # 1. 投影到教师模型维度 (768 -> 1024)
        x = self.input_projection(segments)
        
        # 2. 添加位置编码
        position_indices = torch.arange(time_steps, device=device) 
        pos_embedding = self.pos_encoder(position_indices).unsqueeze(1).expand(-1, batch_size, -1)
        
        x = x + pos_embedding
        x = self.dropout(x)
        
        # 3. 运行 Transformer 编码器
        # src_key_padding_mask: 告诉 Transformer 忽略哪些元素（填充部分）
        output = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        return output
    


    def load_xlarge_weights(self, xlarge_path: str = HUGE_XLARGE_PATH):
            """
            理由：实现 SegmentEncoder 的初始化，使用 HuBERT XLarge 的深层权重
            （第 6 到 17 层，共 12 层），提供强大的语义编码能力。
            """
            print(f"Initializing SegmentEncoder from HuBERT XLarge layers 6-17 at: {xlarge_path}...")
            
            try:
                # 1. 加载预训练的 HuBERT XLarge 模型 (仅用于提取权重)
                # 使用 local_files_only=True 确保从本地路径加载
                hubert_xlarge = HubertModel.from_pretrained(xlarge_path, local_files_only=True)

                # 2. 定义源层索引
                START_LAYER = 6
                END_LAYER = START_LAYER + self.transformer_encoder.num_layers # 6 + 12 = 18

                # 3. 遍历 SegmentEncoder 的层并复制权重
                for i in range(self.transformer_encoder.num_layers):
                    target_layer = self.transformer_encoder.layers[i]
                    
                    # 源层索引: 从 HuBERT 的第 6 层开始
                    source_layer_index = START_LAYER + i 
                    source_layer = hubert_xlarge.encoder.layer[source_layer_index] # 注意：HubertModel 使用 .layer 属性
                    
                    # 复制状态字典
                    target_layer.load_state_dict(source_layer.state_dict())
                
                print(f"SegmentEncoder 的 {self.transformer_encoder.num_layers} 层已成功使用 HuBERT XLarge 的 {START_LAYER} 到 {END_LAYER-1} 层权重初始化。")
                
                # 4. 删除 HuBERT 模型以释放内存
                del hubert_xlarge
                torch.cuda.empty_cache() # 释放 GPU 显存
                
            except Exception as e:
                print(f"FATAL ERROR: Failed to load XLarge weights from {xlarge_path}. Exception: {e}")
                print("Please ensure the HuggingFace HubertModel can load the local path.")
        
