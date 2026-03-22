import torch
import torch.nn as nn
import torch.nn.functional as F
from .BoundaryLoss import BoundaryLoss
class TICSLossCriterion(nn.Module):
    def __init__(self, 
                 pos_weight=15.0, 
                 label_smoothing_kernel=5, 
                 alpha=1.0,  # Boundary 权重
                 beta=1.0,   # MoCo 权重
                 gamma=0.1,  # MSE 权重
                 lambda_k=0.5, # Count (pred_k) 权重
                 temp=0.07):
        super().__init__()
        # 1. 物理边界损失
        self.boundary_loss_fn = BoundaryLoss(
            pos_weight=pos_weight, 
            label_smoothing_kernel=label_smoothing_kernel
        )
        
        # 2. 基础配置
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda_k = lambda_k
        self.temp = temp
        
        # 3. 回归损失器
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss() # 预测数量通常用 L1 更平滑

    def forward(self, model_output, batch, queue):
        """
        model_output: TICS_MoCo 的输出字典
        batch: 含有 'y_boundary', 'mask', 'target_k' 的 batch 数据
        queue: 模型内部维护的负样本队列 [512, K]
        """
        device = model_output['p_score'].device
        
        # --- [1] 物理边界损失 (Boundary) ---
        p_score = torch.sigmoid(model_output['p_score']) 
        loss_bnd = self.boundary_loss_fn(p_score, batch['y_boundary'], batch['mask'])

        # --- [2] 全局对齐损失 (MoCo InfoNCE) ---
        q = model_output["q"]       # [B, 512]
        k = model_output["k_m"]     # [B, 512]
        
        # 正样本
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # 负样本 (从传入的 queue 中拿)
        l_neg = torch.einsum('nc,ck->nk', [q, queue.detach()]) 
        
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temp
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=device)
        loss_moco = F.cross_entropy(logits, labels)

        # --- [3] 语义脑补损失 (MSE) ---
        pred_1024 = model_output["audio_global_pred"]
        target_1024 = model_output["text_global_online"]
        if target_1024 is None:
            # 制造一个全 0 的 target，但不需要梯度
            target_1024 = torch.zeros_like(pred_1024).detach()
            # 设置一个 flag，标记这次 loss 无效
            valid_mask = 0.0
        else:
            valid_mask = 1.0

        # 始终计算 MSE，保持计算图连接
        raw_mse = self.mse_loss(pred_1024, target_1024.detach())
        # 如果无效，乘以 0，梯度阻断；如果有效，乘以 1
        loss_mse = raw_mse * valid_mask    
    

        # --- [4] 精炼计数损失 (Count Loss) ---
        # 预测的 k (pred_k) vs 真实的句子数量 (n_sentences)
        # 假设 batch['n_sentences'] 是 [B] 的长整型 Tensor
        target_k = batch['target_k'].float()
        loss_count = self.l1_loss(model_output['pred_k'], target_k)
        
        # --- [5] 总损失聚合 ---
        total_loss = (self.alpha * loss_bnd) + \
                     (self.beta * loss_moco) + \
                     (self.gamma * loss_mse) + \
                     (self.lambda_k * loss_count)

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"[Warning] NaN loss detected! Masking to 0.")
            total_loss = total_loss * 0.0
            
        return {
            "loss": total_loss,
            "loss_boundary": loss_bnd,
            "loss_moco": loss_moco,
            "loss_mse": loss_mse,
            "loss_count": loss_count
        }