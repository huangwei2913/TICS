import torch
import torch.nn as nn

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