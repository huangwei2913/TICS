import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import HubertModel
import math


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0).unsqueeze(0), persistent=False)

    def forward(self, x):
        # x shape: [Batch, Heads, Seq, Head_Dim]
        seq_len = x.shape[-2]
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        
        x1 = x[..., :self.dim//2]
        x2 = x[..., self.dim//2:]
        rotated = torch.cat((-x2, x1), dim=-1)
        return (x * cos) + (rotated * sin)




class RoPETransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=4096, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        # 独立投影层，方便权重加载
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
        
        # 1. 映射并转换为 [Batch, Heads, Seq, Head_Dim]
        q = self.q_proj(x).view(S, B, self.nhead, self.head_dim).permute(1, 2, 0, 3)
        k = self.k_proj(x).view(S, B, self.nhead, self.head_dim).permute(1, 2, 0, 3)
        v = self.v_proj(x).view(S, B, self.nhead, self.head_dim).permute(1, 2, 0, 3)
        
        # 2. 注入旋转位置信息
        q = self.rope(q)
        k = self.rope(k)
        
        # 3. 注意力计算
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if src_key_padding_mask is not None:
            # mask 为 True 的地方填 -inf
            attn_weights = attn_weights.masked_fill(src_key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_out = torch.matmul(attn_probs, v)
        
        # 4. 转换回 [Seq, Batch, Dim]
        attn_out = attn_out.permute(2, 0, 1, 3).contiguous().view(S, B, D)
        
        # 5. 残差与 MLP
        x = residual + self.dropout(self.out_proj(attn_out))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x



class EnhancedSegmentEncoder(nn.Module):
    def __init__(self, input_dim=1024, segment_dim=1024, num_layers=12):
        super().__init__()
        self.segment_dim = segment_dim
        
        # 将 Hubert-Base 的 768 投影到 1024
        self.input_projection = nn.Linear(input_dim, segment_dim)
        # 注入时长补偿信息
        self.duration_embed = nn.Linear(1, segment_dim)
        
        # 自定义层序列
        self.layers = nn.ModuleList([
            RoPETransformerEncoderLayer(d_model=segment_dim, nhead=16) 
            for _ in range(num_layers)
        ])

    def forward(self, segments, durations, padding_mask):
        """
        segments: [Seq, Batch, 768]
        durations: [Seq, Batch, 1]
        padding_mask: [Batch, Seq] (True 为 padding)
        """
        x = self.input_projection(segments)
        x = x + self.duration_embed(durations)
        
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=padding_mask)
            
        return x # 输出 [Seq, Batch, 1024]

    def load_pretrained_weights(self, large_checkpoint_path="/mnt/facebook/hubert-large-ls960-ft", start_layer=6):
        """
        精准加载 Hubert-Large-LS960-FT 的权重到增强型架构
        """
        from transformers import HubertModel
        print(f"Loading weights from {large_checkpoint_path} starting at layer {start_layer}...")
        
        official_model = HubertModel.from_pretrained(large_checkpoint_path)
        src_layers = official_model.encoder.layers if hasattr(official_model.encoder, 'layers') else official_model.encoder.layer
        
        target_sd = self.state_dict()
        
        # 映射关系字典
        mapping = {
            'q_proj.weight': 'attention.q_proj.weight', 'q_proj.bias': 'attention.q_proj.bias',
            'k_proj.weight': 'attention.k_proj.weight', 'k_proj.bias': 'attention.k_proj.bias',
            'v_proj.weight': 'attention.v_proj.weight', 'v_proj.bias': 'attention.v_proj.bias',
            'out_proj.weight': 'attention.out_proj.weight', 'out_proj.bias': 'attention.out_proj.bias',
            'norm1.weight': 'layer_norm.weight', 'norm1.bias': 'layer_norm.bias',
            'norm2.weight': 'final_layer_norm.weight', 'norm2.bias': 'final_layer_norm.bias',
            'mlp.0.weight': 'feed_forward.intermediate_dense.weight', 'mlp.0.bias': 'feed_forward.intermediate_dense.bias',
            'mlp.2.weight': 'feed_forward.output_dense.weight', 'mlp.2.bias': 'feed_forward.output_dense.bias'
        }

        for i in range(len(self.layers)):
            source_idx = start_layer + i
            src_sd = src_layers[source_idx].state_dict()
            for tgt_suffix, src_suffix in mapping.items():
                tgt_key = f"layers.{i}.{tgt_suffix}"
                if tgt_key in target_sd:
                    target_sd[tgt_key].copy_(src_sd[src_suffix])
        
        self.load_state_dict(target_sd, strict=False)
        print("✅ Pretrained weights injected successfully.")




