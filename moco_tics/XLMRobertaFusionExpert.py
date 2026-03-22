import torch
import torch.nn as nn
from transformers import XLMRobertaModel

class XLMRobertaFusionExpert(nn.Module):
    def __init__(self, xlmr_path, target_dim=1024):
        super().__init__()
        # 加载 768 维的 Base 模型
        self.model = XLMRobertaModel.from_pretrained(xlmr_path, output_hidden_states=True)
        # 冻结
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 挑选 4 层进行融合 (索引对应 hidden_states 的 [4, 7, 10, 13])
        self.layers_to_use = [3, 6, 9, 12] 
        
        # 1. 维度对齐层：从 768 升维到 1024
        # 这样文本导师输出的特征就能直接和语音侧的 1024 维进行 Loss 计算
        self.dim_alignment = nn.Sequential(
            nn.Linear(768, target_dim),
            nn.LayerNorm(target_dim)
        )
        
        # 2. 跨层融合权重
        self.layer_weights = nn.Parameter(torch.ones(len(self.layers_to_use)))
        


    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            # all_layers 长度为 13 (Embedding + 12 layers)
            all_layers = outputs.hidden_states
            # 提取目标层的 [CLS] token
            # hidden_states[i] 形状: [B, T, 768] -> [:, 0, :] 得到 [B, 768]
            cls_list = [all_layers[i][:, 0, :] for i in self.layers_to_use]
            stacked_cls = torch.stack(cls_list, dim=0) # [4, B, 768]

        # 层加权融合
        weights = torch.softmax(self.layer_weights, dim=0).view(-1, 1, 1)
        fused_768 = (stacked_cls * weights).sum(dim=0) # [B, 768]
        
        # 升维到 1024
        fused_1024 = self.dim_alignment(fused_768) # [B, 1024]
          
        return fused_1024