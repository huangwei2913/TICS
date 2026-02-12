import torch
import torch.nn as nn
import torch.nn.functional as F
from .BoundaryLoss import BoundaryLoss
class TICSLossCriterion(nn.Module):
    def __init__(self, 
                 pos_weight=15.0, 
                 label_smoothing_kernel=5, 
                 alpha=1.0,  # Boundary 权重
                 beta=1000.0,   # Distillation 权重
                 gamma=1.0,  # Count 权重
                 delta=0.8): # Emotion 权重
        super().__init__()
        # 1. 初始化你设计的物理边界损失
        self.boundary_loss_fn = BoundaryLoss(
            pos_weight=pos_weight, 
            label_smoothing_kernel=label_smoothing_kernel
        )
        
        # 2. 其他基础损失函数
        self.count_loss_fn = nn.SmoothL1Loss(reduction='mean')
        self.mse_loss_fn = nn.MSELoss(reduction='mean')
        
        # 权重配置
        self.weights = {'alpha': alpha, 'beta': beta, 'gamma': gamma, 'delta': delta}

    def compute_emotion_consistency(self, anchors, target_emo, mask):
        """
        [情感一致性损失] 基于样本内部对齐
        anchors: [B, S, 1024] - 模型精炼出的语义锚点
        target_emo: [B, T, 1024] - emotion2vec 真值特征
        mask: [B, T] - 物理帧掩码
        """
        # 计算音频全局情感基调 (样本内均值)
        # target_emo_global: [B, 1024]
        denom = mask.sum(1, keepdim=True) + 1e-8
        target_emo_global = (target_emo * mask.unsqueeze(-1)).sum(1) / denom
        
        # 计算当前所有锚点的语义平均表示
        # anchor_summary: [B, 1024]
        anchor_summary = anchors.mean(dim=1) 
        
        # 样本内对齐: 确保语义锚点捕获了原始语音的情感精髓
        loss_emo = (1 - F.cosine_similarity(anchor_summary, target_emo_global, dim=-1, eps=1e-5)).mean()
        return loss_emo

    def forward(self, model_output, batch):
        """
        model_output: TICS_MoCo forward 的输出
        batch: Dataset/Collate 的输出
        """
        # 获取基础信息
        mask = batch['mask'] # [B, T]
        
        # --- A. 物理边界损失 (使用你设计的 BoundaryLoss) ---
        # p_score 应该是 Sigmoid 后的概率，如果模型输出是 Logits，请确保经过了 Sigmoid
        p_score = torch.sigmoid(model_output['p_score']) 
        loss_bnd = self.boundary_loss_fn(p_score, batch['y_boundary'], mask)

        # 阶段检查：如果没有文本导师（推理或 Stage 1），直接返回
        if model_output["text_global"] is None:
            return loss_bnd, {"total": loss_bnd.item(), "bnd": loss_bnd.item()}

        # --- B. 全局语义蒸馏损失 (脑补对齐) ---
        audio_g = model_output['audio_global']
        text_g = model_output['text_global']
        T = 0.1 

        # 归一化特征
        audio_g_norm = F.normalize(audio_g, p=2, dim=-1)
        text_g_norm = F.normalize(text_g, p=2, dim=-1)
        cos_sim = (audio_g_norm * text_g_norm).sum(dim=-1) / T
        loss_dist = (1.0 - torch.sigmoid(cos_sim)).mean()
        loss_mse = F.mse_loss(audio_g, text_g)
        loss_dist = (1.0 - F.cosine_similarity(audio_g, text_g, dim=-1)).mean() + 2.0 * loss_mse
  

        # --- C. 篇章结构损失 (Count 回归) ---
        loss_count = self.count_loss_fn(model_output['pred_k'], batch['target_k'])

        # --- D. 情感一致性损失 (基于 emotion2vec 对齐) ---
        # loss_emo = self.compute_emotion_consistency(
        #     model_output['anchors'], 
        #     batch['target_emo'], 
        #     mask
        # )
        loss_emo = torch.tensor(0.0, device=loss_bnd.device, dtype=loss_bnd.dtype)
        # --- 总损失聚合 ---
        total_loss = (self.weights['alpha'] * loss_bnd + 
                      self.weights['beta'] * loss_dist + 
                      self.weights['gamma'] * loss_count )
        #+ 
         #             self.weights['delta'] * loss_emo)

        return total_loss, {
            "total": total_loss.item(),
            "bnd": loss_bnd.item(),
            "dist": loss_dist.item(),
            "count": loss_count.item(),
            "emo": 0.0,
            "pred_k_avg": model_output['pred_k'].mean().item() # 方便监控预测偏好
        }