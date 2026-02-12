import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import HubertModel
import math

#这个模块将变长的段落虚拟成一个“文本句子”的全局描述
class GlobalLatentSummarizer(nn.Module):
    def __init__(self, dim=1024, num_heads=8):
        super().__init__()
        # 这是一个可学习的“提问者”，它代表了模型对“全局文本感”的某种预设
        self.latent_query = nn.Parameter(torch.randn(1, 1, dim)) 
        
        # 使用标准的多头注意力和残差结构
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.ln = nn.LayerNorm(dim)
        
        # 最后的非线性投影，进一步提炼“文本感”
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x, mask=None):
        """
        x: [B, S, 1024] - 经过 Encoder 后的段落序列
        mask: [B, S] - 掩码
        """
        batch_size = x.shape[0]
        # 广播查询向量到 batch size
        q = self.latent_query.expand(batch_size, -1, -1) # [B, 1, 1024]
        
        # 让“提问者”去音频序列里找答案
        # key = value = x
        attn_out, _ = self.attn(q, x, x, key_padding_mask=mask)
        
        # 残差与投影
        out = self.ln(attn_out.squeeze(1)) # [B, 1024]
        out = out + self.mlp(out)
        return out