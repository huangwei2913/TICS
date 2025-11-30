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
                attention_mask: torch.Tensor = None) -> List[torch.Tensor]:
        
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
                    
        return [features_dict[i] for i in layers_to_extract]
    

class SequenceAttentionBlock(nn.Module):
    """
    标准的 Transformer 编码器块，用于融合多层 CLS Token。
    """
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        # 简化 Dropout 和 Add & Norm 步骤
        self.drop = nn.Dropout(0.1) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L_num, C) 
        q = k = v = self.norm1(x)
        # Self-Attention
        attn_output, _ = self.attn(q, k, v)
        x = x + self.drop(attn_output)
        
        # MLP
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x

# --- 3. Fusion & Attention ---
class FeatureFusion(nn.Module):
    def __init__(self, dim, layers_to_use: list, num_heads=8):
        super().__init__()
        self.layers_to_use = layers_to_use
        self.dim = dim # 768
        num_layers = len(layers_to_use)

        # 1. 序列特征融合：可学习的加权求和
        # w: (num_layers) 学习每层序列特征的权重
        self.sequence_weights = nn.Parameter(torch.ones(num_layers, dtype=torch.float32))
        self.norm_sequence = nn.LayerNorm(dim) # 融合后的序列特征做 LayerNorm

        # 2. CLS Token 提取
        # 为每个选定的层创建一个 CrossAttentionBlock 来提取其 CLS Token
        self.attention_blocks = nn.ModuleList([
            CrossAttentionBlock(dim, num_heads=num_heads)
            for _ in layers_to_use
        ])
        
        # 3. CLS Token 融合：使用自注意力机制
        # 一个或多个 Transformer Block，用于融合 L_num 个 CLS Tokens
        self.cls_fusion_block = SequenceAttentionBlock(dim, num_heads) 
        
        # 4. 融合后的 CLS Token 降维层 (可选，如果维度不变，可以不用)
        # 这里仅使用 LayerNorm 来规范化最终的 CLS Token
        self.norm_cls = nn.LayerNorm(dim)

    def forward(self, features_dict: Dict[int, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features_dict: {层索引: (B, T, C) 张量}
        Returns:
            fused_sequence_features (B, T, C)
            fused_cls_token (B, C) <- 修正为 2D，因为 TICSBoundaryStudent 会 unsqueeze
        """
        batch_size = next(iter(features_dict.values())).shape[0]
        time_len = next(iter(features_dict.values())).shape[1]
        
        # 确保 CLS Token 模板的设备和 dtype 与权重一致
        device = self.sequence_weights.device
        dtype = self.norm_sequence.weight.dtype # 使用模型权重精度 (通常是 FP16)
        
        cls_token_template = torch.zeros(batch_size, 1, self.dim, device=device, dtype=dtype)
        
        all_cls_tokens = []
        all_sequence_features = []

        # 1. 提取 CLS Tokens 和收集序列特征
        for i, layer_idx in enumerate(self.layers_to_use):
            sequence_features = features_dict[layer_idx] # (B, T, C)
            
            # 将序列特征转换为融合所需的精度 (FP16/FP32)
            if sequence_features.dtype != dtype:
                 sequence_features = sequence_features.to(dtype)
            
            all_sequence_features.append(sequence_features)
            
            # 拼接: [CLS_token; Sequence_Features] -> (B, T+1, C)
            x_with_cls = torch.cat([cls_token_template.to(sequence_features.dtype), sequence_features], dim=1)
            
            # 运行 CrossAttentionBlock，提取新的 CLS Token (B, 1, C)
            new_cls_token = self.attention_blocks[i](x_with_cls)
            all_cls_tokens.append(new_cls_token.squeeze(1)) # (B, C)

        # --- 第一项改进：融合序列特征 ---
        
        # 堆叠序列特征: List[(B, T, C)] -> (L_num, B, T, C)
        stacked_sequences = torch.stack(all_sequence_features, dim=0) 
        
        # 可学习的加权求和
        # 确保权重求和为 1，使用 softmax 实现归一化
        normalized_weights = torch.softmax(self.sequence_weights, dim=0) # (L_num)
        
        # 广播权重并加权求和: (L_num) * (L_num, B, T, C) -> (B, T, C)
        # unsqueeze 维度使其与 stacked_sequences 对齐: (L_num, 1, 1, 1)
        weighted_sum = (normalized_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * stacked_sequences).sum(dim=0)
        
        fused_sequence_features = self.norm_sequence(weighted_sum) # (B, T, C)

        # --- 第二项改进：融合 CLS Tokens ---
        
        # 堆叠 CLS Tokens: List[(B, C)] -> (B, L_num, C)
        stacked_cls_tokens = torch.stack(all_cls_tokens, dim=1)
        
        # 使用 SequenceAttentionBlock 进行融合
        fused_cls_tokens = self.cls_fusion_block(stacked_cls_tokens) # (B, L_num, C)
        
        # 取融合后的序列的第一个 token 作为最终的全局特征 (类似 Transformer 的 CLS Token 策略)
        final_cls_token = fused_cls_tokens[:, 0] # (B, C)
        
        # 规范化并返回
        fused_cls_token = self.norm_cls(final_cls_token) # (B, C)
        
        # 返回: 融合序列特征 (B, T, C) 和 融合 CLS Token (B, C)
        # 注意: 按照 TICSBoundaryStudent 的最新实现，我们返回 2D (B, C)
        return fused_sequence_features, fused_cls_token

    

import torch
import torch.nn as nn

class TICSBoundaryStudent(nn.Module):
    """
    TICS Student 边界预测网络：
    1. 使用局部序列特征 (seq_feat) 作为 Bi-LSTM 的输入。
    2. 使用全局融合特征 (fused_cls) 来初始化 Bi-LSTM 的初始隐藏状态 (h0, c0)。
    3. Bi-LSTM 捕获上下文后，通过 FC 层映射到边界概率。
    """
    def __init__(self, input_dim=768, hidden_dim=256, dropout=0.1, num_lstm_layers=8):
        """
        Args:
            input_dim (int): 序列特征 (seq_feat) 和全局特征 (fused_cls) 的维度 (如 HuBERT 的 768)。
            hidden_dim (int): LSTM 隐藏状态的维度 (如 256)。
            dropout (float): LSTM 层间 Dropout 率。
            num_lstm_layers (int): LSTM 的层数。
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.num_directions = 2 # 双向 LSTM
        
        # 1. Bi-LSTM 网络
        # 输入维度为 seq_feat 的维度 (768)，不再是 2*768
        self.lstm = nn.LSTM(
            input_size=input_dim,          # 768
            hidden_size=hidden_dim,        # 256
            num_layers=num_lstm_layers,    # 2
            batch_first=True, 
            bidirectional=True, 
            dropout=dropout
        )
        
        # 2. 初始状态投影层 (用于状态初始化)
        # 将 fused_cls (input_dim=768) 投影到 h0 和 c0 的维度 (hidden_dim=256)
        # 每个状态需要一个独立的线性层
        self.cls_h0_projection = nn.Linear(input_dim, hidden_dim)
        self.cls_c0_projection = nn.Linear(input_dim, hidden_dim)
        
        # 3. 预测头 (FC 层)
        # 输入维度: Bi-LSTM 的输出 (hidden_dim * 2) 
        # 输出维度: 边界 Logits (1)
        self.fc = nn.Linear(hidden_dim * self.num_directions, 1)
        
        # 4. 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, seq_feat: torch.Tensor, fused_cls: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq_feat: (B, T, D) - 局部序列特征
            fused_cls: (B, D) - 全局融合特征 (从 FeatureFusion 传入，已优化为 2D)
            
        Returns:
            probs: (B, T) - 边界概率分数
        """
        # 0. 扁平化 LSTM 参数以优化性能 (解决 'not part of single contiguous chunk' 警告)
        self.lstm.flatten_parameters()
        
        B, T, D = seq_feat.shape
        
        # --- 1. 初始化隐藏状态 (h0, c0) ---
        
        # a. 投影 fused_cls: (B, D) -> (B, hidden_dim)
        # 确保数据类型匹配 (DeepSpeed/FP16 兼容性)
        target_dtype = self.cls_h0_projection.weight.dtype
        if fused_cls.dtype != target_dtype:
             fused_cls = fused_cls.to(target_dtype)
             
        h0_base = self.cls_h0_projection(fused_cls) 
        c0_base = self.cls_c0_projection(fused_cls)
        
        # b. 扩展状态以匹配 LSTM 期望的形状
        # 期望形状: [num_layers * num_directions, B, hidden_dim]
        required_layers = self.num_lstm_layers * self.num_directions
        
        h0 = h0_base.unsqueeze(0).repeat(required_layers, 1, 1)
        c0 = c0_base.unsqueeze(0).repeat(required_layers, 1, 1)

        # --- 2. Bi-LSTM 前向传播 ---
        # 传入 seq_feat 和初始化状态
        # lstm_out: (B, T, hidden_dim * 2)
        lstm_out, _ = self.lstm(seq_feat, (h0, c0))
        
        # --- 3. 生成 Logits 和 概率 ---
        # logits: (B, T, 1)
        logits = self.fc(lstm_out)
        
        # probs: (B, T, 1) -> (B, T)
        probs = self.sigmoid(logits)
        
        return probs.squeeze(-1)


class SCPCBoundaryHardener(nn.Module):
    """
    将软概率转换为硬边界 (0/1)，同时使用 Straight-Through Estimator (STE) 允许梯度回传。
    """
    def __init__(self):
        super().__init__()
        # Tanh 用于缩放输入，使其更接近 -1/1 或 0/1 的阶跃形态
        self.tanh = nn.Tanh()
        
    def forward(self, p_score: torch.Tensor) -> torch.Tensor:
        """
        Args:
            p_score: (B, T) 范围 [0, 1] 的概率
        Returns:
            b_hard_ste: (B, T) 范围 {0, 1} 的二值边界 (Tensor)
        """
        
        # 1. 计算软边界 (Soft Boundary)
        # 将概率放大，使其稍微“陡峭”一些，但仍可微
        # 这里的系数 10.0 是经验值
        b_soft = self.tanh(10.0 * p_score)
        
        # 2. 计算硬边界 (Hard Boundary) - 仅用于前向传播
        # 使用极大的系数 (1000.0) 模拟阶跃函数，或者是直接 round
        # 这里使用 tanh 模拟是为了保持数值范围一致性
        # 在数值上，tanh(1000 * x) 对正数极其接近 1，对 0 接近 0
        b_hard = self.tanh(1000.0 * p_score)
        
        # 也可以简单地使用阈值 0.5：
        # b_hard = (p_score > 0.5).float()
        
        # 3. Straight-Through Estimator (STE)
        # 前向传播使用 b_hard (离散值)
        # 反向传播使用 b_soft (梯度)
        # 公式: result = soft + (hard - soft).detach()
        # 前向看: soft + hard - soft = hard
        # 反向看: grad(soft) + 0 = grad(soft)
        b_hard_ste = b_soft + (b_hard - b_soft).detach()
        
        # 确保输出是一个单纯的 Tensor
        return b_hard_ste

def segment_pooling(sequence_features: torch.Tensor, hard_boundaries) -> List[torch.Tensor]:
    """
    Args:
        sequence_features: (B, T, D)
        hard_boundaries:   Expected (B, T) Tensor, but handling defensive cases.
    """
    batch_size, time_steps, dim = sequence_features.shape
    segmented_batch = []
    


    # --- 核心修复 1: 强制处理 Tuple/List 输入 ---
    if isinstance(hard_boundaries, (tuple, list)):
        # 如果不小心传进来了 (Tensor,) 这种元组，取第一个元素
        if len(hard_boundaries) == 1:
            hard_boundaries = hard_boundaries[0]
        else:
            # 如果是真正的列表结构，尝试堆叠
            try:
                hard_boundaries = torch.stack(hard_boundaries)
            except:
                raise TypeError(f"hard_boundaries is a tuple/list of length {len(hard_boundaries)}, expected Tensor.")

    # --- 核心修复 2: 形状检查 ---
    # 确保 hard_boundaries 是 Tensor
    if not isinstance(hard_boundaries, torch.Tensor):
         raise TypeError(f"hard_boundaries expected Tensor, got {type(hard_boundaries)}")

    # 确保 hard_boundaries 形状匹配 (B, T)
    # 如果形状是 (B, T, 1)，squeeze 掉最后一维
    if hard_boundaries.dim() == 3 and hard_boundaries.shape[-1] == 1:
        hard_boundaries = hard_boundaries.squeeze(-1)
        
    # 如果 batch size 不匹配 (极其罕见，可能是 transpose 错误)
    if hard_boundaries.shape[0] != batch_size:
        raise ValueError(f"Batch size mismatch: feats {batch_size}, bounds {hard_boundaries.shape[0]}")

    # ==========================================
    # 下面是之前修复过的 Pooling 逻辑 (保持不变)
    # ==========================================
    for b in range(batch_size):
        seq = sequence_features[b] 
        bounds = hard_boundaries[b] # 现在这里绝对安全了
        
        # 使用 .flatten() 确保一维
        boundary_indices = torch.nonzero(bounds).flatten().long().tolist()
        
        segment_points = [0] + [idx + 1 for idx in boundary_indices]
        
        if not segment_points or segment_points[-1] < time_steps:
             segment_points.append(time_steps)
        
        if segment_points[-1] > time_steps:
             segment_points[-1] = time_steps

        segment_vectors = [] 
        for i in range(len(segment_points) - 1):
            start = segment_points[i]
            end = segment_points[i+1]
            if end > start:
                segment = seq[start:end] 
                pooled_vector = segment.mean(dim=0) 
                segment_vectors.append(pooled_vector)
        
        if not segment_vectors:
             if time_steps > 0:
                 segment_vectors.append(seq.mean(dim=0))
             else:
                 segment_vectors.append(torch.zeros(dim, device=seq.device, dtype=seq.dtype))
            
        segmented_batch.append(torch.stack(segment_vectors, dim=0))

    return segmented_batch