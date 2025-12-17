import argparse
import deepspeed
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from moco_tics.model import TICS_MoCo
from moco_tics.data_loader import TICSDataset, tics_collate_fn
# 假设您的项目中有这些工具类


class BoundaryLoss(nn.Module):
    def __init__(self, pos_weight=15.0):
        super().__init__()
        # 边界通常是极少数点，使用 pos_weight 缓解正负样本不平衡
        self.pos_weight = pos_weight

    def forward(self, P_score, Y_true, mask=None):
        """
        P_score: (B, T)
        Y_true: (B, T)
        mask: (B, T) 1.0 为真实语音，0.0 为 Padding
        """
        # 对齐长度
        min_t = min(P_score.size(1), Y_true.size(1))
        P_score = P_score[:, :min_t]
        Y_true = Y_true[:, :min_t].float()
        
        # 计算加权 BCE
        # pos_weight > 1 强制模型关注那些稀疏的 '1' (边界点)
        loss = F.binary_cross_entropy(P_score, Y_true, reduction='none')
        
        if self.pos_weight != 1.0:
            weight = 1.0 + Y_true * (self.pos_weight - 1.0)
            loss = loss * weight

        if mask is not None:
            mask = mask[:, :min_t].float()
            loss = (loss * mask).sum() / (mask.sum() + 1e-6)
        else:
            loss = loss.mean()
            
        return loss


class TICSContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, outputs):
        # 这里的 q1, k2 形状是 (S_max, B, D)
        loss_a = self._compute_batch_loss(outputs['q1'], outputs['k2'])
        loss_b = self._compute_batch_loss(outputs['q2'], outputs['k1'])
        return (loss_a + loss_b) / 2

    def _compute_batch_loss(self, q, k):
        # 1. 均值池化，将 (S_max, B, D) 转换为 (B, D) 的句子级别表征
        # 注意：这里推荐先做均值池化再做对比，这在 MoCo 中更稳定
        # 如果你想做 Segment-level 对比，需要极其复杂的对齐，Stage I 建议做 Sequence-level
        q_avg = q.mean(dim=0) # (B, D)
        k_avg = k.mean(dim=0) # (B, D)
        
        # 2. 计算余弦相似度矩阵
        logits = torch.matmul(q_avg, k_avg.T) / self.temperature # (B, B)
        
        # 3. 对角线是正样本
        labels = torch.arange(q_avg.shape[0], device=q.device)
        return F.cross_entropy(logits, labels)


def parse_args():
    parser = argparse.ArgumentParser(description="TICS Stage I Training")
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lambda_sup', type=float, default=1.0)
    parser.add_argument('--lambda_moco', type=float, default=0.5)
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. 模型初始化
    # 注意：TEACHER_CONFIG 需匹配你 EnhancedSegmentEncoder 的初始化参数
    TEACHER_CONFIG = {
        'input_dim': 768,       # HuBERT base 输出维度
        'segment_dim': 1024,    # 我们要加载 XLarge 权重，所以这里是 1024
        'num_layers': 12,
        'dropout': 0.1
    }
    
    model = TICS_MoCo(
        backbone_path="/mnt/facebook/hubert-base-ls960", 
        teacher_config=TEACHER_CONFIG
    )
    
    # ⚠️ 关键步骤：加载 HuBERT XLarge 权重到我们的 RoPE Encoder 中
    # 假设你的 TICS_MoCo 类中已经集成了这个方法
    model.encoder_q.load_xlarge_weights("/mnt/facebook/hubert-xlarge-ls960-ft")
    model.encoder_k.load_xlarge_weights("/mnt/facebook/hubert-xlarge-ls960-ft")

    # 2. 数据加载
    train_dataset = TICSDataset(csv_path=args.csv_path)
    
    # 3. DeepSpeed 初始化
    model_engine, optimizer, trainloader, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        collate_fn=tics_collate_fn,
    )

    # 4. 损失函数定义
    # 使用我们优化后的向量化版本
    contrastive_criterion = TICSContrastiveLoss(temperature=0.1).to(model_engine.device)
    boundary_criterion = BoundaryLoss(pos_weight=15.0).to(model_engine.device)

    model_engine.train()
    for epoch in range(args.epochs):
        for step, batch in enumerate(trainloader):
            # 将数据移动到 GPU 并根据需要转半精度
            view1 = batch["view1"].to(model_engine.device).half()
            view2 = batch["view2"].to(model_engine.device).half()
            y_true = batch["y_true"].to(model_engine.device)
            # 音频掩码用于 Boundary Loss (B, T)
            # 假设 tics_collate_fn 返回了有效长度信息
            y_mask = batch.get("y_mask", torch.ones_like(y_true)).to(model_engine.device)

            # --- 前向传播 ---
            # 接收包含 z, p_score, mask 等的字典
            outputs = model_engine(view1, view2)

            # --- 计算损失 ---
            # 1. 边界监督损失 (使用 View 1 的预测概率)
            loss_sup = boundary_criterion(outputs["P_score"], y_true, mask=y_mask)

            # 2. MoCo 对比损失 (跨视图语义对齐)
            loss_moco = contrastive_criterion(outputs)

            # 3. 总损失加权
            total_loss = args.lambda_moco * loss_moco + args.lambda_sup * loss_sup

            # --- 反向传播 ---
            model_engine.backward(total_loss)
            model_engine.step()

            # --- 日志打印 ---
            if step % 10 == 0 and args.local_rank <= 0:
                # 还可以打印 P_score 的均值来观察模型是否在“偷懒”全预测为0
                p_mean = outputs["P_score"].mean().item()
                print(f"Epoch {epoch} | Step {step} | Loss: {total_loss.item():.4f} | "
                      f"MoCo: {loss_moco.item():.4f} | Sup: {loss_sup.item():.4f} | P_avg: {p_mean:.4f}")

        # 保存每轮检查点
        if args.local_rank <= 0:
            model_engine.save_checkpoint(save_dir="checkpoints_stage1", tag=f"epoch_{epoch}")

if __name__ == "__main__":
    main()