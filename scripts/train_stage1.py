import argparse
import deepspeed
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from moco_tics.model import TICS_MoCo
from moco_tics.data_loader import TICSDataset, tics_collate_fn
from tqdm import tqdm
import os
from moco_tics.TicsAugmentation import TicsAugmentation
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


# 确保 Y_true 和 P_score 的长度对齐
class BoundaryLoss(nn.Module):
    def __init__(self, pos_weight=15.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, P_score, Y_true, mask=None):
        # 确保时间步对齐
        min_t = min(P_score.size(1), Y_true.size(1))
        P_score = P_score[:, :min_t]
        Y_true = Y_true[:, :min_t].float()
        Y_true = Y_true.to(P_score.dtype)
        if mask is not None:
            mask = mask.to(P_score.dtype)
        # BCE 损失
        loss = F.binary_cross_entropy(P_score, Y_true, reduction='none')
        
        # 类别不平衡处理 (边界点非常稀疏，所以 pos_weight 设为 15)
        if self.pos_weight != 1.0:
            weight = 1.0 + Y_true * (self.pos_weight - 1.0)
            loss = loss * weight
            
        # 掩码处理：只计算音频实际长度部分的损失
        if mask is not None:
            mask = mask[:, :min_t].float()
            return (loss * mask).sum() / (mask.sum() + 1e-6)
        return loss.mean()

class TICSContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, outputs):
        # 计算 q1->k2 和 q2->k1 的对称损失
        loss_a = self._compute_segment_loss(outputs['q1'], outputs['k2'], outputs['mask1'])
        loss_b = self._compute_segment_loss(outputs['q2'], outputs['k1'], outputs['mask2'])
        return (loss_a + loss_b) / 2

    def _compute_segment_loss(self, q, k, mask):
            """
            高性能、高强度版本：
            1. 修复长度不匹配 Bug (s_min)
            2. 实现分段级对比学习 (Flatten Contrast)
            """
            # [维度转换] 从 (S, B, D) 变为 (B, S, D)
            q, k = q.transpose(0, 1), k.transpose(0, 1)
            
            # [Bug 修复] 强制对齐序列长度，防止 441 vs 440 报错
            s_min = min(q.size(1), k.size(1), mask.size(1))
            q, k, mask = q[:, :s_min, :], k[:, :s_min, :], mask[:, :s_min]

            # [上强度核心] 展平所有 Batch 里的有效片段
            # valid_indices 是一个 (B, S) 的布尔矩阵
            valid_indices = ~mask 
            
            # 提取有效片段：结果维度为 (N_total_valid_segments, D)
            # 这一步会自动把 Batch 维度和 Seq 维度压平，只留下真实的语音向量
            q_valid = q[valid_indices] 
            k_valid = k[valid_indices]

            # 安全检查
            if q_valid.size(0) == 0:
                return torch.tensor(0.0, device=q.device, requires_grad=True)

            # [计算相似度矩阵] (N_total, D) @ (D, N_total) -> (N_total, N_total)
            # 每个片段都要和 Batch 内所有其他片段做对比
            logits = torch.matmul(q_valid, k_valid.T) / self.temperature
            
            # [构造标签] 对角线位置即为正样本（对应的片段）
            labels = torch.arange(q_valid.size(0), device=q.device)

            # 返回 CrossEntropy Loss
            return F.cross_entropy(logits, labels)
        




def parse_args():
    parser = argparse.ArgumentParser(description="TICS Stage I Training")
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lambda_sup', type=float, default=1.0)
    parser.add_argument('--lambda_moco', type=float, default=0.5)
    parser.add_argument('--aug_mode', type=str, default="shuffle")
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. 路径与配置准备
    checkpoint_dir = "checkpoints_stage1"
    best_model_dir = os.path.join(checkpoint_dir, "best")
    if args.local_rank <= 0:
        os.makedirs(checkpoint_dir, exist_ok=True)

    TEACHER_CONFIG = {
        'input_dim': 768, 
        'segment_dim': 1024, 
        'num_layers': 12,
    }

    augmentor = TicsAugmentation(mode=args.aug_mode)

    # 2. 模型初始化
    # TICS_MoCo 内部会自动处理 encoder_k 的初始化和 load_pretrained_weights
    model = TICS_MoCo(
        backbone_path="/mnt/facebook/hubert-base-ls960", 
        teacher_config=TEACHER_CONFIG
    )

    for param in model.backbone.parameters():
        param.requires_grad = False

    # 3. 数据准备
    train_dataset = TICSDataset(csv_path=args.csv_path, augmentor=augmentor, stage=1)

    # 4. DeepSpeed 初始化
    # model_engine 处理分布式训练、FP16 混合精度和优化器更新
    model_engine, optimizer, trainloader, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,  # ✅ 恢复这个！
        collate_fn=tics_collate_fn,
    )
    

    # 5. 损失函数定义
    contrastive_criterion = TICSContrastiveLoss(temperature=0.1).to(model_engine.device)
    boundary_criterion = BoundaryLoss(pos_weight=15.0).to(model_engine.device)

    # 6. 训练监控变量
    best_loss = float('inf')
    
    # --- 训练循环 ---
    for epoch in range(args.epochs):
        model_engine.train()
        epoch_moco_loss = 0.0
        epoch_sup_loss = 0.0
        epoch_total_loss = 0.0
        
        # 使用 tqdm 在主进程显示进度
        pbar = tqdm(trainloader, desc=f"Epoch {epoch}", disable=(args.local_rank > 0))
        
        for step, batch in enumerate(pbar):
            # 获取数据并转换为模型所需的半精度
            view1 = batch["view1"].to(model_engine.device).half()
            view2 = batch["view2"].to(model_engine.device).half()
            y_true = batch["y_true"].to(model_engine.device)
            y_mask = batch.get("y_mask", torch.ones_like(y_true)).to(model_engine.device)

            # --- Forward ---
            #outputs = model_engine(view1, view2)
            outputs = model_engine(view1, view2, aug_mode=args.aug_mode)
            # --- Loss 计算 ---
            # 1. 边界发现损失 (监督学习)
            loss_sup = boundary_criterion(outputs["P_score"], y_true, mask=y_mask)
            
            # 2. 语义对比损失 (MoCo 自监督)
            loss_moco = contrastive_criterion(outputs)

            # 总损失加权
            total_loss = args.lambda_moco * loss_moco + args.lambda_sup * loss_sup

            # --- Backward & Optimize ---
            model_engine.backward(total_loss)
            model_engine.step()

            # 累积统计量
            epoch_total_loss += total_loss.item()
            epoch_moco_loss += loss_moco.item()
            epoch_sup_loss += loss_sup.item()

            # --- 实时监控逻辑 ---
            if step % 10 == 0 and args.local_rank <= 0:
                p_avg = outputs["P_score"].mean().item()
                # 检查 P_avg 是否异常（例如全部趋向 0 或 1），预防模型崩塌
                status_msg = "OK" if 0.01 < p_avg < 0.5 else "WARNING: COLLAPSE?"
                
                pbar.set_postfix({
                    "Loss": f"{total_loss.item():.4f}",
                    "MoCo": f"{loss_moco.item():.4f}",
                    "P_avg": f"{p_avg:.3f}",
                    "Status": status_msg
                })

        # --- Epoch 结束：模型保存与最优逻辑 ---
        avg_loss = epoch_total_loss / len(trainloader)
        
        if args.local_rank <= 0:
            print(f"\n>> Epoch {epoch} Finished. Average Loss: {avg_loss:.4f}")
            
            # 1. 保存最新的 Checkpoint (DeepSpeed 格式)
            model_engine.save_checkpoint(checkpoint_dir, tag=f"epoch_{epoch}")
            
            # 2. 最优模型替代逻辑
            if avg_loss < best_loss:
                print(f"🏆 New Best Loss: {avg_loss:.4f} (Previous: {best_loss:.4f})")
                best_loss = avg_loss
                
                # 保存 DeepSpeed 格式的最优模型
                model_engine.save_checkpoint(checkpoint_dir, tag="best")
                
                # 同时额外保存一份标准的 PyTorch 权重，方便 Stage 2 直接调用
                best_pt_path = os.path.join(checkpoint_dir, "tics_stage1_best.pt")
                torch.save(model.state_dict(), best_pt_path)
                print(f"✅ Best weights synced to {best_pt_path}")

    print("Training Stage 1 completed.")

if __name__ == "__main__":
    main()