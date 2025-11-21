# moco_tics/backbone.py (修改版)

import torch
import torch.nn as nn
from transformers import HubertModel
from typing import List, Dict

class FrozenHubertBackbone(nn.Module):
    # output_layer 现在用于单层模式（兼容旧代码），layers_to_extract 用于多层模式
    def __init__(self, model_path: str = None): 
        super().__init__()
        
        # 定义本地路径
        if model_path is None:
            local_path = "/mnt/facebook/hubert-base-ls960" 
        else:
            local_path = model_path
            
        print(f"Loading Backbone from local path: {local_path}...")

        # from_pretrained 会自动识别本地文件夹
        # **注意：** 确保 HuggingFace Transformer 库版本较新，可以处理 output_hidden_states=True
        self.model = HubertModel.from_pretrained(local_path)
        
        # 核心操作：冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False
            
    def forward(self, wav_input: torch.Tensor, 
                layers_to_extract: List[int] = None, 
                attention_mask: torch.Tensor = None) -> Dict[int, torch.Tensor]:
        
        # wav_input: (Batch, Time_samples)
        
        # 如果没有指定层，则默认提取最后一层（例如 Layer 12）
        if layers_to_extract is None:
             layers_to_extract = [12] 
        
        with torch.no_grad():
            outputs = self.model(
                wav_input, 
                attention_mask=attention_mask, 
                output_hidden_states=True
            )
            
            # hidden_states 是一个 tuple，包含 (embedding, layer_1, ..., layer_12)
            # 所以 Layer N 对应 outputs.hidden_states[N]
            hidden_states = outputs.hidden_states 
            
            # 提取所需的特征并放入字典
            features_dict: Dict[int, torch.Tensor] = {}
            for layer_idx in layers_to_extract:
                if 0 <= layer_idx < len(hidden_states):
                    # 提取特定层 (Batch, Frames, 768)
                    features_dict[layer_idx] = hidden_states[layer_idx]
                else:
                    raise IndexError(f"Layer index {layer_idx} out of range (0 to {len(hidden_states) - 1})")
                    
        return features_dict