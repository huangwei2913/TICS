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

from moco_tics.backbone import FrozenHubertBackbone
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
    
    # 8. 运行学生模型
    print("Running TICS Boundary Student...")
    boundary_probs = student_model(sequence_features_for_boundary, fused_cls)
    
    # --- 9. 最终验证 ---
    print("\n--- Final Verification Results ---")
    
    # a. 序列特征形状 (Fusion Output)
    expected_seq_shape = (BATCH_SIZE, TIME_FRAMES, DIMENSION)
    actual_seq_shape = sequence_features_for_boundary.shape
    assert actual_seq_shape == expected_seq_shape
    print(f"Sequence Features for Boundary Shape: {actual_seq_shape} (PASS)")
    
    # b. 融合 CLS Token 形状 (Fusion Output)
    expected_cls_shape = (BATCH_SIZE, DIMENSION)
    actual_cls_shape = fused_cls.shape
    assert actual_cls_shape == expected_cls_shape
    print(f"Fused Global CLS Token Shape: {actual_cls_shape} (PASS)")
    
    # c. 学生模型输出形状 (Student Output) 🎯
    expected_prob_shape = (BATCH_SIZE, TIME_FRAMES)
    actual_prob_shape = boundary_probs.shape
    assert actual_prob_shape == expected_prob_shape
    print(f"Student Boundary Probs Shape: {actual_prob_shape} (PASS) ✅")

    # d. 设备验证
    assert boundary_probs.device == device
    print(f"Output Device: {device} (PASS)")
    
    print("\n🎉 End-to-End Dataflow Test (HuBERT -> Fusion -> Student) passed successfully!")


if __name__ == "__main__":
    # 请确保 Mlp, DropPath, CrossAttention, CrossAttentionBlock 的定义在运行环境中可用
    # 否则需要先将这些代码粘贴到 test_fusion_integrated.py 的顶部
    test_integration()