import argparse
import deepspeed
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from moco_tics.model import TICS_MoCo # 修正导入路径
from moco_tics.data_loader import TICSDataset, tics_collate_fn # 导入 TICS 数据集和 collate 函数
from util.optimizer import LARS # 假设 LARS 已实现或在 DeepSpeed 配置中定义

# --- 1. 损失函数实现 (Placeholder for now) ---
# TICS 损失函数接受 TICS_MoCo 的输出字典
class TICSContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, outputs):
        """
        outputs: TICS_MoCo 返回的字典，包含归一化后的 p_q1, z_k2 等
        """
        loss_a = self._compute_loss(outputs['q1'], outputs['k2'], outputs['mask_q1'], outputs['len_q1'], outputs['len_k2'])
        loss_b = self._compute_loss(outputs['q2'], outputs['k1'], outputs['mask_q2'], outputs['len_q2'], outputs['len_k1'])
        
        return (loss_a + loss_b) / 2

    def _compute_loss(self, pred, target, mask_pred, len_pred, len_target):
        """
        pred: (T_max, B, D) - Query
        target: (T_max, B, D) - Key
        mask_pred: (B, T_max) - 真实部分为 1 (False in typical padding mask logic, let's adjust)
        """
        # 调整维度为 (B, T, D) 以便批处理
        pred = pred.transpose(0, 1)   # (B, T_q, D)
        target = target.transpose(0, 1) # (B, T_k, D)
        
        batch_size = pred.shape[0]
        total_loss = 0
        valid_batches = 0

        # 由于 View 1 和 View 2 的段数可能不同 (T_q != T_k)
        # 且 TICS 是时序分割，我们需要找到最佳匹配或假设时间对齐。
        # 简化版：我们只计算在有效长度内的 Segment 对比。
        # 为了防止形状不匹配，我们截断到两者的最小长度。
        
        for b in range(batch_size):
            # 获取单个样本的有效长度
            l_p = len_pred[b]
            l_t = len_target[b]
            min_l = min(l_p, l_t)
            
            if min_l == 0: continue

            # 取出有效且对齐的段
            # (min_l, D)
            curr_pred = pred[b, :min_l] 
            curr_target = target[b, :min_l]

            # === InfoNCE 计算 ===
            
            # 1. 计算 Logits (余弦相似度 / Temp)
            # (min_l, D) @ (D, min_l) -> (min_l, min_l)
            # 对角线是正样本 (第 i 段 vs 第 i 段)
            logits = torch.matmul(curr_pred, curr_target.T) / self.temperature
            
            # 2. 生成标签 (对角线是 0, 1, 2...)
            labels = torch.arange(min_l, device=pred.device)
            
            # 3. 计算 Cross Entropy
            # 这会自动计算 Softmax 和 -Log
            loss = self.cross_entropy(logits, labels)
            
            total_loss += loss
            valid_batches += 1

        if valid_batches > 0:
            return total_loss / valid_batches
        else:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

def parse_args():
    parser = argparse.ArgumentParser(description="TICS Training with DeepSpeed")
    parser.add_argument('--local_rank', type=int, default=-1, help='DeepSpeed local rank')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--data_root', type=str, required=True, help='Root directory for the datasets (e.g., /mnt)')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the Aishell-1 metadata CSV file')
    # DeepSpeed 参数
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. 初始化模型
    TEACHER_CONFIG = {
        'input_dim': 768,       # HuBERT Base/Large 特征维度
        'segment_dim': 1024,    # TICS Segment Encoder 维度
        'num_layers': 12,       # TICS Segment Encoder 层数
        'dropout': 0.1
    } 
    # ⚠️ 确保 HuBERT Base 路径正确
    model = TICS_MoCo(backbone_path="/mnt/facebook/hubert-base-ls960", teacher_config=TEACHER_CONFIG)

    # 2. 准备数据 (使用用户提供的路径参数)
    train_dataset = TICSDataset(csv_path=args.csv_path, data_root=args.data_root)
    
    # 3. DeepSpeed 初始化
    # ⚠️ DeepSpeed 需要 collate_fn 传递给 initialize
    model_engine, optimizer, trainloader, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        collate_fn=tics_collate_fn, # 确保使用自定义的 collate_fn
        #config="ds_config.json" # 配置文件路径
    )
    
    # 初始化损失函数 (确保温度系数与 MoCo/TICS 配置一致)
    #contrastive_loss = TICSContrastiveLoss(temp=model.temp).to(model_engine.device)
    contrastive_loss = TICSContrastiveLoss(temperature=model.temp).to(model_engine.device)    
    # 4. 训练循环
    for epoch in range(args.epochs):
        for step, batch in enumerate(trainloader):
            # 获取数据 (DeepSpeed 会自动移动到 GPU)
            view1, view2 = batch
            view1 = view1.half() 
            view2 = view2.half()
            view1 = view1.to(model_engine.device)
            view2 = view2.to(model_engine.device)
            
            # 前向传播
            outputs = model_engine(view1, view2)
            
            # 计算损失 (使用实现的 Loss 模块)
            loss = contrastive_loss(outputs) 
            
            # 反向传播 (DeepSpeed 特有写法)
            model_engine.backward(loss)
            
            # 权重更新 (DeepSpeed 特有写法)
            model_engine.step()
            
            # 日志打印 (仅在 Rank 0)
            if step % 10 == 0 and args.local_rank == 0:
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")
        
        # 保存 Checkpoint
        if args.local_rank == 0:
              model_engine.save_checkpoint(save_dir="checkpoints", tag=f"epoch_{epoch}")

if __name__ == "__main__":
    main()