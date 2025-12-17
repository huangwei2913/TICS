import torch
import torch.nn as nn
import os
import sys
from typing import List, Dict

# -----------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')
sys.path.insert(0, project_root)
# -----------------------------------------------------

from moco_tics.modules import FrozenHubertBackbone
from util.utils import CrossAttentionBlock

class FeatureFusion(nn.Module):
    # 假设这里的实现与上一个回复中的 FeatureFusion 完全相同
    def __init__(self, dim, layers_to_use: List[int], num_heads=8):
        super().__init__()
        self.layers_to_use = layers_to_use
        self.dim = dim # 768

        self.attention_blocks = nn.ModuleList([
            CrossAttentionBlock(dim, num_heads=num_heads)
            for _ in layers_to_use
        ])
        
        in_features = len(layers_to_use) * dim
        out_features = dim 
        self.fusion_projection = nn.Linear(in_features, out_features)
        
    def forward(self, features_dict: Dict[int, torch.Tensor]):
        batch_size = next(iter(features_dict.values())).shape[0]
        device = next(iter(features_dict.values())).device
        
        cls_token_template = torch.zeros(batch_size, 1, self.dim, device=device)

        all_cls_tokens = []
        
        for i, layer_idx in enumerate(self.layers_to_use):
            sequence_features = features_dict[layer_idx]
            x_with_cls = torch.cat([cls_token_template, sequence_features], dim=1)
            new_cls_token = self.attention_blocks[i](x_with_cls) 
            all_cls_tokens.append(new_cls_token)

        fused_cls_tokens = torch.cat(all_cls_tokens, dim=1) 
        fused_cls_tokens_flat = fused_cls_tokens.view(batch_size, -1) 
        fused_cls_token = self.fusion_projection(fused_cls_tokens_flat)
        
        sequence_features_for_boundary = features_dict[self.layers_to_use[-1]] 

        return sequence_features_for_boundary, fused_cls_token



class TICSBoundaryStudent(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, dropout=0.1):
        super().__init__()
        
        # 拼接后的输入维度 = 768 (序列) + 768 (全局CLS) = 1536
        self.combined_input_dim = input_dim * 2 
        
        # 1. Bi-LSTM 1
        self.bi_lstm_1 = nn.LSTM(
            input_size=self.combined_input_dim, 
            hidden_size=hidden_dim, 
            num_layers=1,
            batch_first=True, 
            bidirectional=True
        )
        
        # 2. Bi-LSTM 2
        self.bi_lstm_2 = nn.LSTM(
            input_size=hidden_dim * 2, 
            hidden_size=hidden_dim, 
            num_layers=1,
            batch_first=True, 
            bidirectional=True
        )
        
        # 3. MLP Head (Tanh style)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.Tanh(), 
            nn.Dropout(dropout),
            
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Dropout(dropout),
            
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, sequence_features: torch.Tensor, fused_cls_token: torch.Tensor):
        B, T, C = sequence_features.shape
        
        # --- 全局语境注入 ---
        global_context_expanded = fused_cls_token.unsqueeze(1).expand(-1, T, -1)
        combined_input = torch.cat([sequence_features, global_context_expanded], dim=-1)
        
        # --- Bi-LSTM 处理 ---
        self.bi_lstm_1.flatten_parameters()
        self.bi_lstm_2.flatten_parameters()
        
        x, _ = self.bi_lstm_1(combined_input)
        x, _ = self.bi_lstm_2(x)
        
        # --- 边界预测 ---
        probs = self.mlp(x).squeeze(-1) # (B, T)
        
        return probs





import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# --- 1. 修正后的 Stop Gradient (sg) ---
class StopGradient(torch.autograd.Function):
    """
    Stop Gradient (sg) 功能：
    前向：返回输入值 (Identity)。
    反向：返回 None，即梯度为 0，阻止梯度流过此路径。
    """
    @staticmethod
    def forward(ctx, input):
        # 前向传播：保留输入值
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播：返回 None，表示输入对输出的梯度为 0。
        return None 
        
# 重新定义 sg 符号，对应 SCPC 论文的 Stop Gradient
sg = StopGradient.apply

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





#到后面我们可以使用attetnion去优化分段池化操作
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





from transformers import HubertModel
import torch
import torch.nn as nn

import torch.nn as nn
import torch.nn.functional as F

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

# =======================================================
# 集成测试函数
# =======================================================
def test_integration():
    print("--- Starting HuBERT Backbone & Fusion Integration Test ---")
    
    # 1. 配置参数
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 2
    SAMPLE_RATE = 16000
    DURATION_SEC = 2
    TIME_SAMPLES = SAMPLE_RATE * DURATION_SEC
    DIMENSION = 768
    LAYERS_TO_EXTRACT = [2, 5, 9] # 您的选择

    # 2. 模拟输入数据
    fake_wav = torch.randn(BATCH_SIZE, TIME_SAMPLES, device=device) 
    print(f"Input WAV shape: {fake_wav.shape}")
    
    # 3. 实例化 HuBERT Backbone
    LOCAL_MODEL_PATH = "/mnt/facebook/hubert-base-ls960" # 您的本地路径
    backbone = FrozenHubertBackbone(model_path=LOCAL_MODEL_PATH).to(device)
    
    # 4. 运行 HuBERT 前向传播
    print(f"Extracting features from layers {LAYERS_TO_EXTRACT}...")
    features_dict = backbone(fake_wav, layers_to_extract=LAYERS_TO_EXTRACT)
    
    # 验证 HuBERT 输出
    first_key = LAYERS_TO_EXTRACT[0]
    first_feature_shape = features_dict[first_key].shape
    TIME_FRAMES = first_feature_shape[1] # 动态获取帧数 (例如 99)
    print(f"HuBERT Layer {first_key} output shape: {first_feature_shape}")
    
    # 5. 实例化特征融合模块
    fusion_model = FeatureFusion(dim=DIMENSION, layers_to_use=LAYERS_TO_EXTRACT).to(device)
    
    # 6. 运行融合模块
    print("Running Cross-Attention Fusion...")
    sequence_features_for_boundary, fused_cls = fusion_model(features_dict)

    # 7. 实例化学生模型
    student_model = TICSBoundaryStudent(input_dim=DIMENSION).to(device)
    hardener_model = SCPCBoundaryHardener().to(device)

    # 8. 运行学生模型
    print("Running TICS Boundary Student...")
    P_score = student_model(sequence_features_for_boundary, fused_cls)
    

    # 9. 运行硬化模块
    print("Running SCPC Boundary Hardener...")
    b_soft, b_hard_ste = hardener_model(P_score)


    # ====================================================
    # 10. SCPC Hardener 专项功能测试 (核心新增部分)
    # ====================================================
    print("\n--- Hardener Functional & STE Test ---")
    
    # Test A: 边界值检查 (验证软硬化效果)
    # 模拟输入 P_score，看 b_soft 和 b_hard_ste 的差异
    fake_p_scores = torch.tensor([[0.01, 0.5, 0.999]], device=device)
    test_b_soft, test_b_hard_ste = hardener_model(fake_p_scores)
    
    # b_soft (tanh(10*P)) 应该显示更柔和的阈值效应
    # b_hard_ste (tanh(1000*P)) 应该显示极硬的二值化效应
    print(f"Test P_scores: {fake_p_scores.squeeze().tolist()}")
    print(f"Test b_soft:   {test_b_soft.squeeze().tolist()}")
    print(f"Test b_hard:   {test_b_hard_ste.squeeze().tolist()}")
    
    # 检查硬边界是否接近 1 (除了极小值 0.01 以外)
    assert test_b_hard_ste[0][1].item() > 0.99999, "硬边界 b_hard_ste 未能对 0.5 输入进行硬化。"
    print("Check 1: 硬边界 b_hard_ste 成功硬化输入 P_score (PASS)")

    # Test B: 梯度流检查 (验证 STE 机制)
    
    # 重新创建一个 P_score 张量，并要求梯度
    P_score_test = torch.tensor([[0.5]], device=device, requires_grad=True)
    
    # 运行硬化模块
    b_soft_test, b_hard_ste_test = hardener_model(P_score_test)
    
    # 假损失 L = sum(b_hard_ste * 2)。梯度 dL/d(b_hard_ste) = 2
    fake_loss = (b_hard_ste_test * 2).sum() 
    
    # 反向传播
    fake_loss.backward()
    
    # 预期梯度 dL/dP = d(b_soft)/dP * dL/d(b_hard_ste) 
    # = [10 * sech^2(10*p)] * 2
    scale = 10.0
    p_val = P_score_test.item()
    expected_grad = 2.0 * scale * (1.0 - torch.tanh(scale * P_score_test)**2) 
    
    actual_grad = P_score_test.grad
    
    # 检查 P_score_test 的梯度 (允许浮点误差)
    is_grad_correct = torch.allclose(actual_grad, expected_grad, atol=1e-5)
    assert is_grad_correct, f"Check 2: STE梯度失败。实际: {actual_grad.item()}, 预期: {expected_grad.item()}"
    print("Check 2: STE 梯度流验证 (PASS)")
    
    print("--- Hardener Functional & STE Test Passed ---")


    # 11. 最终验证 (形状验证)
    print("\n--- Final Verification Results ---")
    
    # c. 学生模型输出形状
    expected_prob_shape = (BATCH_SIZE, TIME_FRAMES)
    assert P_score.shape == expected_prob_shape
    print(f"Student P_score Shape: {P_score.shape} (PASS)")
    
    # d. 边界输出形状 (新增)
    assert b_soft.shape == expected_prob_shape
    assert b_hard_ste.shape == expected_prob_shape
    print(f"Soft Boundary b_soft Shape: {b_soft.shape} (PASS)")
    print(f"Hard Boundary b_hard_ste Shape: {b_hard_ste.shape} (PASS) ✅")

    # e. 设备验证
    assert b_hard_ste.device == device
    print(f"Output Device: {device} (PASS)")
    
    print("\n🎉 End-to-End Dataflow Test (HuBERT -> Fusion -> Student -> Hardener) passed successfully!")


    from torch.nn.utils.rnn import pad_sequence

    # ====================================================
    # 11. Segment Pooling 专项测试 (新增)
    # ====================================================
    print("\n--- Segment Pooling Functional & Differentiability Test ---")

    # 准备模拟数据 (Batch Size=1)
    TEST_SEQ_LEN = 10
    TEST_DIM = 5
    # 确保特征需要梯度，以便后续检查梯度是否能流回
    fake_seq_features = torch.arange(
        TEST_SEQ_LEN * TEST_DIM, 
        dtype=torch.float, 
        device=device
    ).view(1, TEST_SEQ_LEN, TEST_DIM)
    fake_seq_features.requires_grad_(True) # <-- 必须要求梯度

    # 模拟硬边界: 在索引 3 和 7 处切分
    # 0 1 2 3| 4 5 6 7| 8 9
    # 边界点 (1) 在索引 3 和 7
    # 边界点+1 (新片段开始) 在索引 4 和 8
    fake_boundaries = torch.tensor([
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0] 
    ], dtype=torch.float, device=device)

    # 1. 执行分段池化
    segmented_list = segment_pooling(
        sequence_features=fake_seq_features, 
        hard_boundaries=fake_boundaries
    )

    # --- Check A: 功能正确性 (分段数量和数值) ---
    assert len(segmented_list) == 1, "分段池化 Batch Size 错误。"
    segments = segmented_list[0] # (Num_Segments, D)

    # 预期分段数量: 3 个片段 (0-3, 4-7, 8-9)
    expected_num_segments = 3
    assert segments.shape[0] == expected_num_segments, \
        f"Check A1: 分段数量错误。预期 {expected_num_segments}, 实际 {segments.shape[0]}"
    print(f"Check A1: 分段数量 {expected_num_segments} (PASS)")

    # 验证第一个片段的均值 (0:4 索引)
    # 对应张量元素 0, 1, 2, 3，每个都是 5 维向量。
    # 验证第一个维度 (D=0) 的均值: (0+5+10+15) / 4 = 7.5
    expected_first_dim_mean = (fake_seq_features[0, 0, 0] + 
                            fake_seq_features[0, 1, 0] + 
                            fake_seq_features[0, 2, 0] + 
                            fake_seq_features[0, 3, 0]) / 4
                            
    assert torch.allclose(segments[0, 0], expected_first_dim_mean, atol=1e-5), \
        f"Check A2: 第一个片段的均值计算错误。预期 {expected_first_dim_mean.item()}, 实际 {segments[0, 0].item()}"
    print("Check A2: 片段均值计算 (PASS)")


    # --- Check B: 可微分性 (梯度流动) ---
    # 假设一个简单的损失: 所有片段向量的总和
    dummy_loss = segments.sum()
    dummy_loss.backward()

    # 验证输入特征的梯度是否为非零
    # 由于分段池化是平均操作，所有被使用的输入帧都应有非零梯度。
    assert fake_seq_features.grad is not None, "Check B1: 梯度对象不存在。"
    assert fake_seq_features.grad.abs().sum().item() > 0, "Check B2: 梯度为零，池化操作不可微分。"

    # 验证梯度分布 (例如，第一个片段 (0:4) 的梯度应该相等)
    expected_grad_value = 1.0 / 4.0 # 1/片段长度 (4)
    actual_grad_value = fake_seq_features.grad[0, 0, 0].item()

    # 允许浮点误差
    is_grad_correct = torch.allclose(
        torch.tensor(actual_grad_value), 
        torch.tensor(expected_grad_value), 
        atol=1e-5
    )
    assert is_grad_correct, \
        f"Check B3: 梯度值错误。预期 {expected_grad_value}, 实际 {actual_grad_value}"
    print("Check B3: 梯度回传与平均操作相符 (PASS) ✅")


    # --- Check C: 填充与 Mask 生成 ---
    # 这一步是为教师模型 E_seg 准备输入

    # 1. 填充到 Batch 张量
    # 必须使用集成测试的实际输出
    final_segmented_list = segmented_list # 这里使用模拟数据，但实际应该使用 segmented_list

    # Pad Sequence 默认返回 (Max_Segments, B, D)
    padded_segments_TBD = pad_sequence(final_segmented_list, batch_first=False) 
    # shape: (3, 1, 5)

    # 2. 创建 Mask
    batch_sizes = [s.shape[0] for s in final_segmented_list]
    max_len = padded_segments_TBD.shape[0]

    # 创建一个充满 True 的布尔张量
    padding_mask = torch.ones(
        (len(batch_sizes), max_len), 
        dtype=torch.bool, 
        device=device
    )

    # 标记真实数据为 False (即不被 Mask)
    for i, length in enumerate(batch_sizes):
        padding_mask[i, :length] = False 
        
    # 验证形状
    assert padded_segments_TBD.shape == (expected_num_segments, 1, TEST_DIM)
    assert padding_mask.shape == (1, expected_num_segments)

    # 验证 Mask 值 (所有都应该是 False，因为只有一个样本且没有填充)
    assert not padding_mask.all().item(), "Check C: 填充 Mask 值错误，不应全为 True。"

    print(f"Check C1: 填充张量形状 {padded_segments_TBD.shape} (PASS)")
    print(f"Check C2: 填充 Mask 形状 {padding_mask.shape} (PASS)")

    print("--- Segment Pooling Test Passed ---")


    # ----------------------------------------------------
    # 12. SegmentEncoder Functional & Differentiability Test
    # ----------------------------------------------------
    print("\n--- SegmentEncoder Test (Teacher Model) ---")

    TEST_D_IN = 768
    TEST_D_OUT = 1024
    TEST_BATCH_SIZE = 2
    TEST_NUM_LAYERS = 4 

    # --- A. 准备模拟 Segment Pooling 输出 (Batch Size = 2) ---

    # 样本 1: 3 个片段 (叶节点)
    segment_1 = torch.randn(3, TEST_D_IN, device=device, requires_grad=True)
    # 样本 2: 5 个片段 (叶节点)
    segment_2 = torch.randn(5, TEST_D_IN, device=device, requires_grad=True)

    # 1. 填充到 Batch 张量
    # Max_Segments = 5。输出形状: (Max_Segments, B, D) -> (5, 2, 768)
    # 注意：需要确保导入了 pad_sequence
    from torch.nn.utils.rnn import pad_sequence 

    padded_segments_TBD = pad_sequence([segment_1, segment_2], batch_first=False) 

    # FIX: 显式告诉 PyTorch 保留这个非叶张量的梯度
    padded_segments_TBD.retain_grad()

    # 2. 创建 Mask
    batch_lengths = [s.shape[0] for s in [segment_1, segment_2]] 
    max_len = padded_segments_TBD.shape[0] 

    # 初始化 Mask (B, T) -> (2, 5)
    padding_mask = torch.ones(
        (TEST_BATCH_SIZE, max_len), 
        dtype=torch.bool, 
        device=device
    )

    # 标记真实数据为 False (即不被 Mask)
    for i, length in enumerate(batch_lengths):
        padding_mask[i, :length] = False 
        
    # --- B. 实例化和前向传播 ---

    # 使用模拟的 SegmentEncoder 
    teacher_encoder = SegmentEncoder(
        input_dim=TEST_D_IN, 
        segment_dim=TEST_D_OUT,
        num_layers=TEST_NUM_LAYERS 
    ).to(device)

    for param in teacher_encoder.parameters():
        param.requires_grad = True

    # 执行前向传播
    try:
        segment_embeddings = teacher_encoder(padded_segments_TBD, padding_mask) # (T, B, D_out)
        
    except Exception as e:
        print(f"Check B: SegmentEncoder 前向传播失败: {e}")
        raise e # 如果失败，直接抛出异常

    # --- C. 检查形状和功能 ---

    expected_shape = (max_len, TEST_BATCH_SIZE, TEST_D_OUT) 
    assert segment_embeddings.shape == expected_shape, \
        f"Check C1: 输出形状错误。预期 {expected_shape}, 实际 {segment_embeddings.shape}"
    print(f"Check C1: 输出形状 {segment_embeddings.shape} (PASS)")

    # 检查填充位置的梯度是否为零 (最严格的检查)
    dummy_loss = segment_embeddings.sum()
    dummy_loss.backward()

    # --- D. 检查梯度流动 ---

    # D1: 检查 Encoder 权重是否有梯度
    encoder_grads = sum([p.grad.abs().sum().item() for p in teacher_encoder.parameters() if p.grad is not None])
    assert encoder_grads > 0, "Check D1: SegmentEncoder 权重梯度为零，不可训练。"
    print("Check D1: 教师模型权重梯度 (PASS) ✅")

    # D2: 检查输入叶张量 (segment_1, segment_2) 的梯度是否非零 (原始输入)
    input_grad_sum = segment_1.grad.abs().sum().item() + segment_2.grad.abs().sum().item()
    assert input_grad_sum > 0, "Check D2: 梯度未流回 Segment Pooling 输出。"
    print("Check D2: 梯度回传到 Segment Pooling 输出 (PASS) ✅")

    # D3: 检查填充位置的梯度是否被屏蔽 (验证 Masking 机制)
    # 样本 1 填充位置的索引 (3, 4)
    # 现在 padded_segments_TBD.grad 已经被 retain_grad() 填充了
    grad_at_padding_sample1 = padded_segments_TBD.grad[3:, 0, :].abs().sum().item() 
    assert torch.isclose(torch.tensor(grad_at_padding_sample1), torch.tensor(0.0), atol=1e-5), \
        f"Check D3: 填充位置梯度非零。实际总和: {grad_at_padding_sample1}"
    print("Check D3: 填充位置梯度被有效屏蔽 (PASS) ✅")

    print("--- SegmentEncoder Test Passed ---")



if __name__ == "__main__":
    # 请确保 Mlp, DropPath, CrossAttention, CrossAttentionBlock 的定义在运行环境中可用
    # 否则需要先将这些代码粘贴到 test_fusion_integrated.py 的顶部
    test_integration()