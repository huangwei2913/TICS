import torch
import torch.nn as nn
from transformers import HubertModel
from util.utils import CrossAttentionBlock

class FeatureFusion(nn.Module):
    def __init__(self, dim, layers_to_use: list, num_heads=8):
        super().__init__()
        self.layers_to_use = layers_to_use
        self.dim = dim # 768

        # 1. 创建 CLS Token (作为查询 Q)
        # B x 1 x C，为每个选中的层创建一个可学习的 CLS token
        # self.cls_tokens = nn.Parameter(torch.zeros(1, 1, dim)) 
        
        # 实际上，由于我们不需要多个 CLS Token 来代表整个序列，
        # 我们可以让 CrossAttentionBlock 自动处理 Q，只需将序列特征传入。
        
        # 2. 为每个选定的层创建一个 CrossAttentionBlock 来提取其 CLS Token
        self.attention_blocks = nn.ModuleList([
            CrossAttentionBlock(dim, num_heads=num_heads)
            for _ in layers_to_use
        ])
        
        # 3. 融合后的降维层
        # 输入维度: layers_to_use 数量 * 768
        in_features = len(layers_to_use) * dim
        # 输出维度: 回到 HuBERT 的 768 维
        out_features = dim 
        
        # 线性层用于拼接和降维
        self.fusion_projection = nn.Linear(in_features, out_features)
        
    def forward(self, features_dict: dict):
        """
        Args:
            features_dict: 包含 HuBERT 多层输出的字典。
                           {层索引: (B, T, C) 张量}
        
        Returns:
            fused_sequence_features (B, T, C)
            fused_cls_tokens (B, 1, C)
        """
        batch_size = next(iter(features_dict.values())).shape[0] # B
        time_len = next(iter(features_dict.values())).shape[1] # T
        
        # 1. 提取序列特征 (B, T, C) 和 CLS Tokens (B, 1, C)
        all_cls_tokens = []
        
        # 注意: 您的 CrossAttentionBlock 是为 (B, N, C) 设计的，
        # 且查询 Q 总是来自 x[:, 0:1, ...]，这意味着我们需要一个 CLS 
        # Token 预先连接到序列特征上。
        
        # 创建一个占位 CLS token (B, 1, C)
        # 注意: 在 HuBERT/Wav2vec2 结构中，通常没有 CLS Token，所以我们必须自己构造
        cls_token_template = torch.zeros(batch_size, 1, self.dim, device=self.fusion_projection.weight.device)

        for i, layer_idx in enumerate(self.layers_to_use):
            sequence_features = features_dict[layer_idx] # (B, T, C)
            
            # 拼接: [CLS_token; Sequence_Features] -> (B, T+1, C)
            x_with_cls = torch.cat([cls_token_template, sequence_features], dim=1)
            
            # 运行 CrossAttentionBlock，返回的是新的 CLS Token (B, 1, C)
            # 因为您的 CrossAttentionBlock 逻辑是: x = x[:, 0:1, ...] + ...
            new_cls_token = self.attention_blocks[i](x_with_cls) 
            
            all_cls_tokens.append(new_cls_token)

        # 2. 融合 CLS Tokens
        # 拼接所有 CLS Tokens: (B, num_layers, C) -> (B, num_layers * C)
        fused_cls_tokens = torch.cat(all_cls_tokens, dim=1) # (B, L_num, C)

        # 降维投影: (B, L_num * C) -> (B, 1, C)
        # 注意: 需要展平 L_num 和 C 维度
        fused_cls_tokens_flat = fused_cls_tokens.view(batch_size, -1) # (B, L_num * C)
        fused_cls_token = self.fusion_projection(fused_cls_tokens_flat).unsqueeze(1) # (B, 1, C)
        
        # 3. 返回融合后的 CLS Token (作为句级别/全局特征)
        # 同时也返回您之前选定的序列特征（例如Layer 9）作为边界检测的输入
        # 我们这里假设边界检测主要依赖于您之前选定的 Layer 9 的序列特征
        sequence_features_for_boundary = features_dict[self.layers_to_use[-1]] # 默认使用最后一层作为序列特征

        return sequence_features_for_boundary, fused_cls_token.squeeze(1) # (B, T, C), (B, C)