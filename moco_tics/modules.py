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
    def __init__(self, model_path: str = None): 
        super().__init__()
        
        # 定义本地路径
        local_path = model_path if model_path is not None else "/mnt/facebook/hubert-base-ls960" 
            
        print(f"Loading Backbone from local path: {local_path}...")

        # 加载预训练模型
        self.model = HubertModel.from_pretrained(local_path)
        
        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False
            
    def forward(self, wav_input: torch.Tensor, 
                layers_to_extract: List[int] = None, 
                attention_mask: torch.Tensor = None) -> Dict[int, torch.Tensor]:
        
        #print(f"layers_to_extract: {layers_to_extract}...")
        # 如果没有指定层，则默认提取最后一层
        if layers_to_extract is None:
             layers_to_extract = [12] 
        
        with torch.no_grad():
            outputs = self.model(
                wav_input, 
                attention_mask=attention_mask, 
                output_hidden_states=True
            )
            
            # hidden_states (embedding, layer_1, ..., layer_12)
            hidden_states = outputs.hidden_states 
            
            features_dict: Dict[str, torch.Tensor] = {} # 改为 str
            for layer_idx in layers_to_extract:
                if 0 <= layer_idx < len(hidden_states):
                    # 将层号转换为字符串，避开 DeepSpeed 的 Key 类型检查
                    features_dict[str(layer_idx)] = hidden_states[layer_idx]
                else:
                    raise IndexError(f"Layer index {layer_idx} out of range")

        #print(f"features_dicts..................: {features_dict}...") 

        return features_dict
                    

    

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
        self.layers_to_use = layers_to_use # 这里通常是 [2, 5, 9] 这样的 int 列表
        self.dim = dim
        num_layers = len(layers_to_use)
        self.sequence_weights = nn.Parameter(torch.ones(num_layers, dtype=torch.float32))
        self.norm_sequence = nn.LayerNorm(dim)
        self.attention_blocks = nn.ModuleList([
            CrossAttentionBlock(dim, num_heads=num_heads)
            for _ in layers_to_use
        ])
        self.cls_fusion_block = SequenceAttentionBlock(dim, num_heads) 
        self.norm_cls = nn.LayerNorm(dim)

    def forward(self, features_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. 获取基础维度信息 (使用 .values() 不受 Key 类型影响)
        first_tensor = next(iter(features_dict.values()))
        batch_size = first_tensor.shape[0]
        time_len = first_tensor.shape[1]
        
        device = self.sequence_weights.device
        dtype = self.norm_sequence.weight.dtype
        
        # 2. 准备容器
        cls_token_template = torch.zeros(batch_size, 1, self.dim, device=device, dtype=dtype)
        all_cls_tokens = []
        all_sequence_features = []
        
        # 3. 核心修改点：遍历并使用 str(layer_idx) 索引
        for i, layer_idx in enumerate(self.layers_to_use):
            # 将 int 类型的 layer_idx 转为字符串，以匹配 Backbone 返回的 Dict Key
            str_key = str(layer_idx)
            
            if str_key not in features_dict:
                raise KeyError(f"Layer {str_key} not found in features_dict. Available: {list(features_dict.keys())}")
            
            sequence_features = features_dict[str_key]
            
            # 确保精度一致 (DeepSpeed 开启 fp16 时非常重要)
            if sequence_features.dtype != dtype:
                 sequence_features = sequence_features.to(dtype)
            
            all_sequence_features.append(sequence_features)
            
            # Cross-Attention 融合
            x_with_cls = torch.cat([cls_token_template, sequence_features], dim=1)
            new_cls_token = self.attention_blocks[i](x_with_cls)
            all_cls_tokens.append(new_cls_token.squeeze(1)) # 取出更新后的 CLS
            
        # 4. 序列特征加权融合 (Weighted Average)
        stacked_sequences = torch.stack(all_sequence_features, dim=0) 
        normalized_weights = torch.softmax(self.sequence_weights, dim=0)
        
        # 修正广播机制权重相乘
        weighted_sum = (normalized_weights.view(-1, 1, 1, 1) * stacked_sequences).sum(dim=0)
        fused_sequence_features = self.norm_sequence(weighted_sum)
        
        # 5. CLS Token 融合 (Sequence Attention)
        stacked_cls_tokens = torch.stack(all_cls_tokens, dim=1) # (B, num_layers, dim)
        fused_cls_tokens = self.cls_fusion_block(stacked_cls_tokens)
        
        # 取出融合后的第一个位置作为 final CLS
        final_cls_token = fused_cls_tokens[:, 0]
        fused_cls_token = self.norm_cls(final_cls_token)
        
        return fused_sequence_features, fused_cls_token
    

import torch
import torch.nn as nn

class TICSBoundaryStudent(nn.Module):
    """
    TICS Student 边界预测网络（优化版）：
    1. LSTM 直接处理 HuBERT 的 768 维原始特征，避免投影导致的特征损失。
    2. residual_proj 仅作为支路，将 768 映射到 1024 以匹配双向 LSTM 的输出维度。
    3. 采用 Pre-activation 思想的残差连接 + LayerNorm。
    """
    def __init__(self, input_dim=768, hidden_dim=512, dropout=0.1, num_lstm_layers=12):
        super().__init__()
        
        self.input_dim = input_dim          # 768
        self.hidden_dim = hidden_dim        # 512
        self.num_lstm_layers = num_lstm_layers
        self.num_directions = 2             # 双向
        self.output_dim = hidden_dim * self.num_directions  # 1024
        
        # 1. Bi-LSTM：输入直接对齐 HuBERT (768)
        self.lstm = nn.LSTM(
            input_size=input_dim,           # 修改回 768
            hidden_size=hidden_dim,         # 512
            num_layers=num_lstm_layers,     # 12
            batch_first=True, 
            bidirectional=True, 
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # 2. 残差投影层：仅用于将输入 768 升维到 1024，以便与 LSTM 输出相加
        self.residual_proj = nn.Linear(input_dim, self.output_dim)
        
        # 3. 归一化与 Dropout
        self.lstm_norm = nn.LayerNorm(self.output_dim)
        self.lstm_dropout = nn.Dropout(dropout)
        
        # 4. 初始状态投影层 (用于将 768 维 fused_cls 转为 LSTM 的 hidden 状态)
        self.cls_h0_projection = nn.Linear(input_dim, hidden_dim)
        self.cls_c0_projection = nn.Linear(input_dim, hidden_dim)
        
        # 5. 预测头 (FC 层)
        self.fc = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim // 2),  # 1024 -> 512
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.output_dim // 2, 1)                  # 512 -> 1
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, seq_feat: torch.Tensor, fused_cls: torch.Tensor):
        """
        seq_feat: (B, T, 768)
        fused_cls: (B, 768)
        """
        self.lstm.flatten_parameters()
        B, T, D = seq_feat.shape
        device = seq_feat.device
        
        # --- 1. 初始化隐藏状态 (h0, c0) ---
        target_dtype = self.cls_h0_projection.weight.dtype
        fused_cls = fused_cls.to(target_dtype)
        
        h0_base = self.cls_h0_projection(fused_cls)  # (B, 512)
        c0_base = self.cls_c0_projection(fused_cls)  # (B, 512)

        required_layers = self.num_lstm_layers * self.num_directions # 24
        h0 = h0_base.unsqueeze(0).repeat(required_layers, 1, 1)  # (24, B, 512)
        c0 = c0_base.unsqueeze(0).repeat(required_layers, 1, 1)  # (24, B, 512)
        
        # --- 2. 前向传播：LSTM 主路 + 投影残差支路 ---
        
        # 2.1 支路：对齐维度 (768 -> 1024)
        residual = self.residual_proj(seq_feat)      # (B, T, 1024)
        
        # 2.2 主路：LSTM 直接处理原始 768 维特征
        # 注意：这里不再需要 input_proj
        lstm_out, _ = self.lstm(seq_feat, (h0, c0))  # (B, T, 1024)
        
        # 2.3 残差融合 + 归一化
        # 将 LSTM 的建模结果与原始输入的映射相加
        E_context = self.lstm_norm(lstm_out + residual)
        E_context = self.lstm_dropout(E_context)
        
        # --- 3. 输出 ---
        logits = self.fc(E_context)                  # (B, T, 1)
        probs = self.sigmoid(logits).squeeze(-1)     # (B, T)
        #用长短期记忆网络（Bi-LSTM）去‘理解’整段语音的起承转合，最后由全连接层（FC）在每一帧做‘是非题’。
        return probs, E_context

#1. 它是如何做“是非题”的？（FC + Sigmoid）在每一帧（时间步 $t$），Bi-LSTM 都会输出一个 1024 维的向量。这个向量里包含了“过去”和“未来”对当前时刻的影响。
# FC 层 (1024 → 512 → 1)：就像是一个过滤器。它在 1024 个特征中寻找那些“能量突变”、“语速放缓”、“基频下降”等信号。
# Sigmoid 函数：将 FC 的输出压缩到 $0$ 到 $1$ 之间。输出 0.9：模型非常有信心这里是一个边界（比如一句话结束了）。
# 输出 0.1：模型认为这里还在发音中，不能切分。2. 为什么需要输出这个“边界概率”？
# 在 MoCo 架构中，这个概率 $P_{score}$ 有两个至关重要的用途：监督学习 (Supervised Loss)：
# 你会拿这个 $P_{score}$ 和真实的标签（CSV 里的标注）做二元交叉熵损失（BCE Loss）。
# 这迫使 Student 网络学会精准定位语音的停顿点。软切分 (Soft Segmentation)：这是最关键的设计逻辑！
# $P_{score}$ 越高的地方，意味着这里越倾向于是一个 Segment 的终点。这些概率将指导后续如何把 HuBERT 的长序列切分成一个个小块（Segments）
# ，送入后面的 Encoder。3. 为什么这个设计是“聪明”的？如果只用简单的卷积或线性层，模型只能看到“局部”。但语音的边界是需要上下文的：Bi-LSTM 的作用：
# 它能看到当前时刻之前和之后的信息。比如：模型发现后面有很长时间的静音（通过向后看），同时发现前面刚结束一个完整的单词（通过向前看），那么它在当前位置输出 $P_{score}=0.98$ 的逻辑就非常稳固。4. 逻辑复盘：特征与概率的“双重产出”注意你的 return 语句：Pythonreturn probs, E_context
#probs (B, T)：是“是非题”的结果，用于算边界 Loss 和指导切分。E_context (B, T, 1024)：
# 是“心路历程”，它不仅仅是概率，还包含了丰富的上下文表征。它会被送入 Encoder，作为 MoCo 对比学习的基础。



#将 $P_{\text{score}}$ 转化为硬边界 $b_{\text{hard\_ste}}$，同时通过 $b_{\text{soft}}$ 路径保留梯度通道。
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




#增加返回持续时间
#
#
#作用1：韵律感知 (Prosody Awareness)：在语音中，长音（强调）和短音（停顿或快速词汇）承载了极强的语义信息。
#Mean Pooling 会抹平波形强度，但 duration 找回了时间维度的厚度
#
def segment_pooling_with_durations(sequence_features: torch.Tensor, hard_boundaries: torch.Tensor):
    """
    全向量化实现：同时返回池化特征和每个段的持续时间（帧数）。
    
    Returns:
        segmented_batch: List[torch.Tensor] - 每个元素 (N_i, D)
        durations_batch: List[torch.Tensor] - 每个元素 (N_i, 1)
    """
    B, T, D = sequence_features.shape
    device = sequence_features.device

    
    # --- Step 1: 生成 Segment ID ---
    shift_boundaries = torch.cat([torch.zeros((B, 1), device=device), hard_boundaries[:, :-1]], dim=1)
    seg_ids = torch.cumsum(shift_boundaries, dim=1).long() # (B, T)

    # --- Step 2: 准备聚合容器 ---
    num_segments = seg_ids.max().item() + 1
    seg_sum = torch.zeros((B, num_segments, D), device=device, dtype=sequence_features.dtype)
    seg_count = torch.zeros((B, num_segments, 1), device=device, dtype=sequence_features.dtype)

    # --- Step 3: 并行聚合 (Scatter Add) ---
    seg_ids_expanded = seg_ids.unsqueeze(-1).expand(-1, -1, D)
    seg_sum.scatter_add_(1, seg_ids_expanded, sequence_features)
    
    # 这里的 seg_count 记录了每个 Segment ID 对应的帧数
    seg_count.scatter_add_(1, seg_ids.unsqueeze(-1), torch.ones((B, T, 1), device=device, dtype=sequence_features.dtype))

    # --- Step 4: 计算平均特征 ---
    seg_avg = seg_sum / torch.clamp(seg_count, min=1.0)

    # --- Step 5: 截断并打包回 List ---
    segmented_batch = []
    durations_batch = []
    
    actual_seg_counts = seg_ids.max(dim=1)[0] + 1
    
    for b in range(B):
        n = actual_seg_counts[b].item()
        
        # 提取该 sample 真实的特征段
        segmented_batch.append(seg_avg[b, :n])
        
        # 提取该 sample 真实的各段长度 (帧数)
        # 注意：这里返回的是原始帧数，后续进入 duration_embed 前可以转为 float
        durations_batch.append(seg_count[b, :n])

    return segmented_batch, durations_batch




#改进快速版本
def segment_pooling_tensorized(sequence_features: torch.Tensor, hard_boundaries: torch.Tensor):
    """
    全向量化实现，无 Python 循环，无 CPU 瓶颈。
    
    Args:
        sequence_features: (B, T, D) - 帧特征 (E_context)
        hard_boundaries: (B, T) - 0/1 边界信号 (来自 Hardener)
    Returns:
        segmented_batch: List[torch.Tensor] - 每个 Tensor 形状为 (N_i, D)
    """
    B, T, D = sequence_features.shape
    device = sequence_features.device

    # --- Step 1: 为每一帧生成唯一的 Segment ID ---
    # 使用 cumsum 确定每一帧属于第几个段
    # 我们希望边界点本身属于当前段的结束，或者下一段的开始。
    # 按照 TICS 逻辑，边界点 i 是段的最后一个元素，所以我们对 boundaries 做 shift
    # shift_boundaries = [0, b_0, b_1, ..., b_{T-1}]
    shift_boundaries = torch.cat([torch.zeros((B, 1), device=device), hard_boundaries[:, :-1]], dim=1)
    seg_ids = torch.cumsum(shift_boundaries, dim=1).long() # (B, T)

    # --- Step 2: 准备聚合容器 ---
    num_segments = seg_ids.max().item() + 1
    # 存储特征累加和
    seg_sum = torch.zeros((B, num_segments, D), device=device, dtype=sequence_features.dtype)
    # 存储每个段包含的帧数（用于求平均）
    seg_count = torch.zeros((B, num_segments, 1), device=device, dtype=sequence_features.dtype)

    # --- Step 3: 并行聚合 (Scatter Add) ---
    # 将 seg_ids 扩展到特征维度 (B, T, D)
    seg_ids_expanded = seg_ids.unsqueeze(-1).expand(-1, -1, D)
    
    # 核心操作：根据 seg_ids 把特征填入对应的段索引中
    seg_sum.scatter_add_(1, seg_ids_expanded, sequence_features)
    # 计算每个段有多少帧
    seg_count.scatter_add_(1, seg_ids.unsqueeze(-1), torch.ones((B, T, 1), device=device, dtype=sequence_features.dtype))

    # --- Step 4: 计算平均特征 ---
    # 避免除以 0 (有些预留的 seg_id 可能在某些 batch 中没用到)
    seg_avg = seg_sum / torch.clamp(seg_count, min=1.0)

    # --- Step 5: 转换回 List 结构以匹配后续代码 ---
    # 因为每个 batch 的段数不同，我们需要根据真实的段数进行截断
    segmented_batch = []
    # 计算每个 sample 真实的段数
    actual_seg_counts = seg_ids.max(dim=1)[0] + 1
    for b in range(B):
        n = actual_seg_counts[b].item()
        segmented_batch.append(seg_avg[b, :n])

    return segmented_batch


#原始版本
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





#(训练时对齐)： 使用 I_true（真值索引）。这通常用于第二阶段，或者作为第一阶段的“锚点”。

def segment_pooling_true(E_frame, I_true, N_seg_max):
    """
    基于地面真值索引 I_true 对帧特征 E_frame 进行 Segment Pooling (求平均)。
    
    Args:
        E_frame (torch.Tensor): [B, N_frame, D_feat] - 编码器帧特征
        I_true (torch.Tensor): [B, N_frame] - 地面真值 Segment 索引 (0, 0, 1, 1, ...)
        N_seg_max (int): 当前 Batch 中最大的 Segment 数量 (M_max)
        
    Returns:
        E_speech_true (torch.Tensor): [B, N_seg_max, D_feat] - 聚合后的 Segment 特征
    """
    B, N_frame, D_feat = E_frame.shape
    device = E_frame.device
    
    # --- 步骤 A: 准备 Aggregation 目标张量 ---
    # target_shape: [B, N_seg_max, D_feat]
    target_shape = (B, N_seg_max, D_feat)
    
    # 初始化求和张量和计数张量
    E_sum = torch.zeros(target_shape, dtype=E_frame.dtype, device=device)
    E_count = torch.zeros(target_shape, dtype=E_frame.dtype, device=device)
    
    # --- 步骤 B: 展平张量以便于 Scatter 操作 ---
    
    # 展平特征: [B * N_frame, D_feat]
    E_frame_flat = E_frame.view(-1, D_feat)
    
    # 展平索引: [B * N_frame]
    I_true_flat = I_true.view(-1)
    
    # 关键：创建 target index。我们需要将 [B*N_frame] 的 I_true_flat 扩展到 [B*N_frame, D_feat] 
    # 形状，以便 scatter_add_ 在 D_feat 维度上进行正确的求和。
    I_true_flat_expanded = I_true_flat.unsqueeze(-1).expand_as(E_frame_flat) # [B*N_frame, D_feat]
    
    # --- 步骤 C: 执行 Scatter Add (求和) ---
    # scatter_add_(dim=0, index, src)
    # target: [B * N_seg_max, D_feat]
    E_sum_flat = E_sum.view(B * N_seg_max, D_feat)
    
    # 1. 计算总和 (Sum): 将 E_frame_flat 的特征加到 E_sum_flat 的指定位置
    # 这里需要一个更复杂的 index 来包含 Batch ID 和 Segment ID
    # 由于 PyTorch 的 scatter 对维度 index 比较严格，这里简化为 Batch 循环或使用 torch_scatter 库
    # 
    # 优化方案：使用 for 循环 (更易理解和实现)
    for b in range(B):
        # 针对当前 Batch 的帧特征和索引
        E_frame_b = E_frame[b]  # [N_frame, D_feat]
        I_true_b = I_true[b]    # [N_frame]
        
        # 1. 求和
        E_sum[b].index_add_(0, I_true_b, E_frame_b) # 将 E_frame_b 按 I_true_b 的索引加到 E_sum[b] 上
        
        # 2. 计数
        # 创建一个全 1 的张量，用于计数
        ones = torch.ones_like(E_frame_b) 
        E_count[b].index_add_(0, I_true_b, ones) 
        
    # --- 步骤 D: 求平均 (Divide by Count) ---
    
    # 避免除以零 (Segment Padding)
    E_count = E_count.clamp(min=1) 
    
    # E_speech_true 即为 E_sum / E_count (元素级除法)
    E_speech_true = E_sum / E_count
    
    return E_speech_true


