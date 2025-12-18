import torch
import torch.nn as nn
from transformers import HubertModel
import torch.nn.functional as F
import os
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')

# ==========================================
# 1. 核心模型组件 (RoPE + EncoderLayer)
# ==========================================

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.float32)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos().to(dtype), persistent=False)
        self.register_buffer('sin_cached', emb.sin().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int = None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        
        # x shape: [Batch, Heads, Seq_Len, Head_Dim]
        seq_len = x.shape[-2]
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0) # [1, 1, Seq, Dim]
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        
        # 旋转逻辑
        x1 = x[..., :self.dim//2]
        x2 = x[..., self.dim//2:]
        rotated = torch.cat((-x2, x1), dim=-1)
        return (x * cos) + (rotated * sin)

class RoPETransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=4096, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rope = RotaryEmbedding(dim=self.head_dim)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model)
        )

    def forward(self, x, src_key_padding_mask=None):
        # x: [Seq, Batch, Dim]
        residual = x
        x = self.norm1(x)
        S, B, D = x.shape
        
        # QKV Projections
        q = self.q_proj(x).view(S, B, self.nhead, self.head_dim).permute(1, 2, 0, 3)
        k = self.k_proj(x).view(S, B, self.nhead, self.head_dim).permute(1, 2, 0, 3)
        v = self.v_proj(x).view(S, B, self.nhead, self.head_dim).permute(1, 2, 0, 3)
        
        # Apply RoPE
        q = self.rope(q)
        k = self.rope(k)
        
        # Self-Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if src_key_padding_mask is not None:
            # mask: [Batch, Seq] -> [Batch, 1, 1, Seq]
            attn_weights = attn_weights.masked_fill(src_key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_out = torch.matmul(attn_probs, v) # [B, H, S, Head_Dim]
        attn_out = attn_out.permute(2, 0, 1, 3).contiguous().view(S, B, D)
        
        x = residual + self.dropout(self.out_proj(attn_out))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x

# ==========================================
# 2. 增强型 SegmentEncoder (包含 load 逻辑)
# ==========================================

class EnhancedSegmentEncoder(nn.Module):
    def __init__(self, input_dim=768, segment_dim=1024, num_layers=12):
        super().__init__()
        self.segment_dim = segment_dim
        self.input_projection = nn.Linear(input_dim, segment_dim)
        self.duration_embed = nn.Linear(1, segment_dim)
        self.layers = nn.ModuleList([
            RoPETransformerEncoderLayer(d_model=segment_dim, nhead=16) 
            for _ in range(num_layers)
        ])

    def forward(self, segments, durations, padding_mask):
        x = self.input_projection(segments)
        x = x + self.duration_embed(durations)
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=padding_mask)
        return x

    def load_pretrained_weights(self, large_checkpoint_path, start_layer=6):
        """
        容错加载 Hubert-Large-LS960-FT 的指定层权重
        """
        if not os.path.exists(large_checkpoint_path):
            logging.error(f"❌ 错误: 路径不存在: {large_checkpoint_path}")
            return

        logging.info(f"正在从 {large_checkpoint_path} 注入权重...")
        official_model = HubertModel.from_pretrained(large_checkpoint_path)
        
        # 探测层属性 (兼容 .layers 或 .layer)
        src_layers = official_model.encoder.layers if hasattr(official_model.encoder, 'layers') else official_model.encoder.layer
        
        target_state_dict = self.state_dict()
        copied_count = 0

        # 精准映射字典 (基于探测结果)
        mapping = {
            'q_proj.weight': 'attention.q_proj.weight',
            'q_proj.bias':   'attention.q_proj.bias',
            'k_proj.weight': 'attention.k_proj.weight',
            'k_proj.bias':   'attention.k_proj.bias',
            'v_proj.weight': 'attention.v_proj.weight',
            'v_proj.bias':   'attention.v_proj.bias',
            'out_proj.weight': 'attention.out_proj.weight',
            'out_proj.bias':   'attention.out_proj.bias',
            'norm1.weight': 'layer_norm.weight',
            'norm1.bias':   'layer_norm.bias',
            'norm2.weight': 'final_layer_norm.weight',
            'norm2.bias':   'final_layer_norm.bias',
            'mlp.0.weight': 'feed_forward.intermediate_dense.weight',
            'mlp.0.bias':   'feed_forward.intermediate_dense.bias',
            'mlp.2.weight': 'feed_forward.output_dense.weight',
            'mlp.2.bias':   'feed_forward.output_dense.bias'
        }

        for i in range(len(self.layers)):
            source_idx = start_layer + i
            target_prefix = f"layers.{i}."
            source_layer = src_layers[source_idx]
            src_sd = source_layer.state_dict()

            for tgt_suffix, src_suffix in mapping.items():
                tgt_key = target_prefix + tgt_suffix
                if src_suffix in src_sd and tgt_key in target_state_dict:
                    target_state_dict[tgt_key].copy_(src_sd[src_suffix])
                    copied_count += 1
        
        self.load_state_dict(target_state_dict, strict=False)
        logging.info(f"✅ 权重注入完成。共成功复制 {copied_count} 个参数张量。")

# ==========================================
# 3. 验证脚本 (修正版)
# ==========================================

def verify_weight_transfer(local_large_path: str):
    logging.info("\n开始权重一致性检查...")
    
    # 1. 初始化你的模型 (假设只取 1 层进行测试)
    my_encoder = EnhancedSegmentEncoder(input_dim=1024, segment_dim=1024, num_layers=1)
    my_encoder.load_pretrained_weights(local_large_path, start_layer=6)
    
    # 2. 加载官方模型
    official_model = HubertModel.from_pretrained(local_large_path)
    # 适配官方层路径
    official_layers = official_model.encoder.layers if hasattr(official_model.encoder, 'layers') else official_model.encoder.layer
    official_layer_6 = official_layers[6]
    
    # 3. 采样对比 Q 矩阵权重
    official_q_weight = official_layer_6.attention.q_proj.weight
    my_q_weight = my_encoder.layers[0].q_proj.weight
    
    print("\n--- 关键参数采样对比 (Layer 6 vs Layer 0) ---")
    print(f"官方 Large (Q[0,0]): {official_q_weight[0,0].item():.6f}")
    print(f"你的模型 (Q[0,0]):   {my_q_weight[0,0].item():.6f}")
    
    diff = torch.abs(official_q_weight - my_q_weight).max().item()
    if diff < 1e-7:
        print(f"✅ 权重数值对齐成功！最大绝对误差: {diff:.2e}")
    else:
        print(f"❌ 权重对齐失败！误差过大: {diff:.2e}")

    # 4. MLP 推理对齐测试
    test_input = torch.randn(1, 1, 1024)
    with torch.no_grad():
        # 官方路径: norm -> intermediate_dense
        official_inter = official_layer_6.feed_forward.intermediate_dense(official_layer_6.layer_norm(test_input))
        # 你的模型路径: norm1 -> mlp[0]
        my_inter = my_encoder.layers[0].mlp[0](my_encoder.layers[0].norm1(test_input))
        
        cos_sim = F.cosine_similarity(official_inter.flatten(), my_inter.flatten(), dim=0)
        print(f"✅ MLP 层中间输出余弦相似度: {cos_sim.item():.6f}")

if __name__ == "__main__":
    # 请确保此路径指向 hubert-large-ls960-ft
    LOCAL_PATH = "/mnt/facebook/hubert-large-ls960-ft"
    verify_weight_transfer(LOCAL_PATH)