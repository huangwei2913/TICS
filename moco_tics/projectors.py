# 文件: projectors.py
import torch
import torch.nn as nn

###为什么要弄： EnhancedSegmentEncoder 输出的是 1024 维的隐变量，
###而 Teacher (emotion2vec+) 输出的是 768 维特征。我们需要一个“适配器”来对齐维度，并输出概率分布以便计算 KL 散度。
class EmotionConsistencyHead(nn.Module):
    def __init__(self, segment_dim=1024, emo_dim=768):
        super().__init__()
        # 瓶颈层设计：先压缩再映射，去除无关噪声
        self.net = nn.Sequential(
            nn.Linear(segment_dim, 512),
            nn.GELU(),
            nn.Linear(512, emo_dim)
        )

    def forward(self, x):
        """
        输出 Log Softmax，配合 F.kl_div 使用更稳定
        """
        logits = self.net(x)
        return logits