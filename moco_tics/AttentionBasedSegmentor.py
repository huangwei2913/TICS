import torch
import torch.nn as nn
import torch.nn.functional as F

#从音频帧pooling到音频段落,这个类纯碎的用于得到分段的，用注意力机制
class AttentiveSegmentPooling(nn.Module):
    def __init__(self, input_dim=1024):
        super().__init__()
        # 这是一个轻量级的注意力打分器
        # 它将 1024 维特征压缩成 1 个“重要性分数”
        self.attn_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1) # 输出标量分数
        )

    def forward(self, sequence_features, hard_boundaries):
        """
        Args:
            sequence_features: (B, T, D)
            hard_boundaries: (B, T) 0/1 边界
        Returns:
            segmented_batch: List[torch.Tensor]
            durations_batch: List[torch.Tensor]
        """
        B, T, D = sequence_features.shape
        device = sequence_features.device

        # --- Step 1: 生成 Segment ID (保持不变) ---
        shift_boundaries = torch.cat([torch.zeros((B, 1), device=device), hard_boundaries[:, :-1]], dim=1)
        seg_ids = torch.cumsum(shift_boundaries, dim=1).long() # (B, T)
        
        # --- Step 2: 计算每一帧的重要性权重 (Attention Weights) ---
        # scores: (B, T, 1)
        # 这里的 scores 代表每一帧的“含金量”
        attn_scores = self.attn_net(sequence_features)
        
        # 为了数值稳定性，我们通常在 softmax 之前做操作
        # 但由于我们是用 scatter 实现段内 softmax，这里用 exp(score) 代替
        # 这种方式实现了 "Segment-wise Softmax"
        attn_weights = torch.exp(attn_scores) # 权重必须为正

        # --- Step 3: 加权特征准备 ---
        # weighted_features: (B, T, D)
        weighted_features = sequence_features * attn_weights

        # --- Step 4: 并行聚合 (Scatter Add) ---
        num_segments = seg_ids.max().item() + 1
        
        # 容器 1: 加权特征和 (分子)
        weighted_sum = torch.zeros((B, num_segments, D), device=device, dtype=sequence_features.dtype)
        # 容器 2: 权重和 (分母) -> 代替了原来的 count
        weights_sum = torch.zeros((B, num_segments, 1), device=device, dtype=sequence_features.dtype)
        # 容器 3: 原始帧数 (仅用于返回 durations)
        seg_frames_count = torch.zeros((B, num_segments, 1), device=device, dtype=sequence_features.dtype)

        # 扩展 seg_ids 以匹配维度
        seg_ids_D = seg_ids.unsqueeze(-1).expand(-1, -1, D)
        seg_ids_1 = seg_ids.unsqueeze(-1)

        # 执行聚合
        weighted_sum.scatter_add_(1, seg_ids_D, weighted_features)
        weights_sum.scatter_add_(1, seg_ids_1, attn_weights)
        seg_frames_count.scatter_add_(1, seg_ids_1, torch.ones_like(attn_weights))

        # --- Step 5: 计算加权平均 (分子 / 分母) ---
        # 这本质上就是：Segment = sum(feat * weight) / sum(weight)
        seg_attentive = weighted_sum / (weights_sum + 1e-8) # 加上 epsilon 防止除零

        # --- Step 6: 打包返回 (保持接口一致) ---
        segmented_batch = []
        durations_batch = []
        actual_seg_counts = seg_ids.max(dim=1)[0] + 1

        for b in range(B):
            n = actual_seg_counts[b].item()
            segmented_batch.append(seg_attentive[b, :n])
            durations_batch.append(seg_frames_count[b, :n])

        return segmented_batch, durations_batch
    
'''

AttentiveSegmentPooling 的引入，实际上是把模型从一个**“基于信号的切分器（Signal Cutter）”** 升维成了一个**“基于语义/情感的理解器（Semantic/Emotional Parser）”**。

它不仅仅是为了让特征更好看，更是为了让模型学会**“为了什么而切分”**。

以下是为什么这个机制能帮你实现**“发现一句完整的、有情感的、有语义的分段”**的底层逻辑：

1. 从“切声音”进化到“切意思” (Semantic Integrity)
传统 Mean Pooling 的问题： 它把所有的帧看作一样重要。如果一段语音里包含了半个字的尾音、一个完整的词、和半个词的起音，平均之后，特征就“糊”了。模型会觉得：“这也没啥明确含义啊，随便切吧。”

Attentive Pooling 的作用： 它会自动寻找**“语义重心（Semantic Center of Gravity）”**。

比如在说“不可思议”时，中间的元音能量大、语义强，Attention 会给高分。

两头的过渡音、或者切分稍微切偏了一点的杂音，Attention 会给低分（自动降噪）。

结果：即使边界切得不是 100% 精确，池化出来的特征依然能代表这个词的核心含义。

反向传播的魔力：因为特征更准了，Stage 2 的对比学习 Loss 就会变低。模型会发现：“原来我把这句完整的话切出来，特征最强，Loss 最低！” 从而倒逼 Stage 1 的边界检测器去寻找那些“语义完整”的切分点。

2. 捕捉情感的“核” (Emotional Coherence)
情感是不均匀的： 一段愤怒的语音，可能只有中间那 0.5 秒的爆发（Pitch 极高、能量极大）体现了愤怒，前后可能是压抑的低沉。

Mean Pooling = 稀释情感： (爆发 + 压抑) / 2 = 平淡。这会导致模型觉得这段话没什么情感，从而无法学会根据情感变化来切分。

Attentive Pooling = 提纯情感： 它会抓住那个“爆发点”。只要这一段里有强情感信号，输出的向量就带有强烈的愤怒特征。 这样，模型就能敏锐地察觉到：“这里情感变了（从平淡变成愤怒），所以我应该在这里切一刀！” —— 这正是你想要的“发现有情感的分段”。

3. 超越“在线会议分割” (Beyond VAD)
一般的在线会议分割（VAD/Speaker Diarization）通常只看：

有没有人说话（能量/静音）。

是谁在说话（声纹）。

而你的 TICS 模型加上这个 Attentive 机制后，能做到：

抗干扰（Robustness）：会议里某人咳嗽了一声，或者敲了一下键盘。Mean Pooling 可能会被这些噪声拉偏特征；但 Attentive Pooling 会学会给这些非语音帧打 0 分。

句法感知（Syntactic Awareness）：它倾向于在一个“意群（Thought Group）”结束时切分，而不是像 VAD 一样一停顿就切。因为只有完整的意群，才能在 XLM-R（文本教师）那边对齐到好的特征。

'''