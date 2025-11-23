import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
from transformers import HubertModel
from util.utils import CrossAttentionBlock

# --- 1. Stop Gradient (核心工具) ---
class StopGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input
    @staticmethod
    def backward(ctx, grad_output):
        return None
sg = StopGradient.apply


# --- 2. Backbone ---
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
    


# --- 3. Fusion & Attention ---

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
    


class TICSBoundaryStudent(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, dropout=0.1):
        """
        Args:
            input_dim (int): HuBERT 特征维度 (默认 768)
            hidden_dim (int): LSTM 隐藏层维度 (默认 256)
            dropout (float): Dropout 概率
        """
        super().__init__()
        
        # ----------------------------------------------------------------
        # 1. 输入维度计算 (Feature Fusion Strategy)
        # ----------------------------------------------------------------
        # 输入包含两部分：
        # 1. 序列特征 (Sequence Features): [B, T, 768]
        # 2. 全局语境 (Fused CLS Token):   [B, 768] -> 扩展后 [B, T, 768]
        # 拼接后的维度 = 768 + 768 = 1536
        self.combined_input_dim = input_dim * 2 
        
        # ----------------------------------------------------------------
        # 2. 双层 Bi-LSTM (Stacked Bi-LSTM)
        # ----------------------------------------------------------------
        # 对应 YAML 中的 bidirectional_1
        self.bi_lstm_1 = nn.LSTM(
            input_size=self.combined_input_dim, 
            hidden_size=hidden_dim, 
            num_layers=1,
            batch_first=True, 
            bidirectional=True
        )
        
        # 对应 YAML 中的 bidirectional_2
        # 输入是上一层的输出 (hidden_dim * 2)，输出保持 hidden_dim
        self.bi_lstm_2 = nn.LSTM(
            input_size=hidden_dim * 2, 
            hidden_size=hidden_dim, 
            num_layers=1,
            batch_first=True, 
            bidirectional=True
        )
        
        # ----------------------------------------------------------------
        # 3. MLP 投影头 (TimeDistributed MLP)
        # ----------------------------------------------------------------
        # 对应 YAML 中的 timedistributed_1 -> timedistributed_2 -> output
        # 结构: Linear -> Tanh -> Linear -> Tanh -> Linear -> Sigmoid
        self.mlp = nn.Sequential(
            # Layer 1
            nn.Linear(hidden_dim * 2, 128),
            nn.Tanh(), # 保留经典的 Tanh 激活
            nn.Dropout(dropout),
            
            # Layer 2
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Dropout(dropout),
            
            # Output Layer
            nn.Linear(64, 1),
            nn.Sigmoid() # 输出概率 [0, 1]
        )

    def forward(self, sequence_features, fused_cls_token):
        """
        Args:
            sequence_features: (Batch, Time, 768) - 来自 HuBERT 最后一层
            fused_cls_token:   (Batch, 768)       - 来自 CrossAttention 融合
            
        Returns:
            probs: (Batch, Time) - 帧级边界概率
        """
        B, T, C = sequence_features.shape
        
        # ====================================================
        # 步骤 1: 全局语境注入 (Global Context Injection)
        # ====================================================
        
        # 1. 将 (B, 768) 的 CLS Token 在时间维度 T 上复制
        # unsqueeze(1) -> (B, 1, 768)
        # expand(-1, T, -1) -> (B, T, 768)
        global_context_expanded = fused_cls_token.unsqueeze(1).expand(-1, T, -1)
        
        # 2. 拼接序列特征和全局特征
        # result -> (B, T, 768 + 768) = (B, T, 1536)
        combined_input = torch.cat([sequence_features, global_context_expanded], dim=-1)
        
        # ====================================================
        # 步骤 2: 时序建模 (Bi-LSTM)
        # ====================================================
        
        # 优化显存布局
        self.bi_lstm_1.flatten_parameters()
        self.bi_lstm_2.flatten_parameters()
        
        # LSTM Layer 1
        # x: (B, T, hidden_dim * 2)
        x, _ = self.bi_lstm_1(combined_input)
        
        # LSTM Layer 2 (Deep Context)
        # x: (B, T, hidden_dim * 2)
        x, _ = self.bi_lstm_2(x)
        
        # ====================================================
        # 步骤 3: 边界预测 (MLP Head)
        # ====================================================
        
        # PyTorch Linear 自动处理 (B, T, Dim)，等同于 TimeDistributed
        probs = self.mlp(x) # -> (B, T, 1)
        
        return probs.squeeze(-1) # -> (B, T)




class SCPCBoundaryHardener(nn.Module):
    def __init__(self, soft_scale=10.0, hard_scale=1000.0):
        super().__init__()
        # SCPC 论文中的固定常数 10 和 1000
        self.soft_scale = soft_scale
        self.hard_scale = hard_scale
        self.tanh = nn.Tanh()

    def forward(self, P_score: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据 SCPC 论文 Equation 5 将原始分数 P_score (p) 转化为可微分的硬边界。

        Args:
            P_score: (B, T) - TICSBoundaryStudent 输出的原始分数 (Logits)。

        Returns:
            bsoft: (B, T) - 用于损失计算的软边界 (梯度通过 bsoft 流动)。
            b_hard_ste: (B, T) - 用于分段切分的硬边界 (带 STE 梯度)。
        """
        
        # 1. 软边界 (bsoft): 用于梯度流动，使用较小的缩放因子
        # bsoft = tanh(10 * p)
        bsoft = self.tanh(self.soft_scale * P_score)
        
        # 2. 极硬边界 (bhard): 接近硬二值化，用于 STE 的前向计算
        # bhard = tanh(1000 * p)
        bhard = self.tanh(self.hard_scale * P_score)
        
        # 3. STE 组合: b = bsoft + sg(bhard - bsoft)
        # 前向： b_hard_ste 的值近似于 bhard (极接近 0 或 1)
        # 反向： 梯度只流经 bsoft 路径 (避免 bhard 路径上的梯度爆炸)
        b_hard_ste = bsoft + sg(bhard - bsoft)
        
        return bsoft, b_hard_ste
    

def segment_pooling(sequence_features: torch.Tensor, hard_boundaries: torch.Tensor) -> List[torch.Tensor]:
    """
    使用硬边界 b_hard_ste 对序列特征进行分段平均池化。

    Args:
        sequence_features: (B, T, D) - 序列特征 (来自 FeatureFusion)。
        hard_boundaries:   (B, T)    - 二值化硬边界 b_hard_ste (0 或 1)。

    Returns:
        List[torch.Tensor]: 包含 Batch 中每个 utterance 的分段特征列表。
                            每个元素是 (Num_Segments, D) 形状的 Tensor。
    """
    batch_size, time_steps, dim = sequence_features.shape
    segmented_batch = []

    for b in range(batch_size):
        seq = sequence_features[b] # (T, D)
        bounds = hard_boundaries[b] # (T)
        
        # 1. 找到边界索引并准备起始/结束点
        # nonzero() 返回索引 (例如，如果 T=99，边界在索引 10, 25, 98)
        boundary_indices = torch.nonzero(bounds).squeeze(-1).tolist()
        
        # 确保起始点是 0
        segment_points = [0] + [idx + 1 for idx in boundary_indices] 
        
        # 确保包含序列的结束点
        if segment_points[-1] < time_steps:
             segment_points.append(time_steps)
        elif segment_points[-1] > time_steps:
             # 处理边界落在序列末尾 T-1 的情况，确保不越界
             segment_points[-1] = time_steps

        # 2. 执行分段和池化
        segment_vectors = [] 
        
        # 遍历所有片段 [start_i : start_{i+1}]
        for i in range(len(segment_points) - 1):
            start = segment_points[i]
            end = segment_points[i+1]
            
            if end > start: # 确保片段长度 > 0
                segment = seq[start:end] 
                
                # Mean Pooling (平均池化) - 可微分
                pooled_vector = segment.mean(dim=0) 
                segment_vectors.append(pooled_vector)
        
        # 3. 极端情况处理: 如果序列中没有检测到边界
        if not segment_vectors and time_steps > 0:
             segment_vectors.append(seq.mean(dim=0))
            
        segmented_batch.append(torch.stack(segment_vectors, dim=0))

    return segmented_batch