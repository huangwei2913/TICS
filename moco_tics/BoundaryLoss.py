import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundaryLoss(nn.Module):
    def __init__(self, pos_weight=10.0, label_smoothing_kernel=5, sigma=1.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.kernel_size = label_smoothing_kernel
        
        if self.kernel_size > 1:
            x = torch.arange(self.kernel_size, dtype=torch.float32) - (self.kernel_size - 1) / 2
            kernel = torch.exp(-0.5 * (x / sigma) ** 2)
            kernel = kernel / kernel.sum()
            self.register_buffer('smooth_kernel', kernel.view(1, 1, -1))
            self.padding = (self.kernel_size - 1) // 2
        else:
            self.smooth_kernel = None

    def forward(self, p_score, y_boundary, mask):
        """
        p_score: [B, T] (Half) - 必须已经过 Sigmoid
        y_boundary: [B, T] (Float)
        """
        # --- 1. 精度与设备对齐 (防止 RuntimeError) ---
        target = y_boundary.to(p_score.device).to(p_score.dtype)
        mask = mask.to(p_score.device).to(p_score.dtype)

        # --- 2. 长度对齐 (防止 ValueError) ---
        min_t = min(p_score.size(1), target.size(1))
        p_score = p_score[:, :min_t]
        target = target[:, :min_t]
        mask = mask[:, :min_t]

        # --- 3. 高斯平滑 (确保卷积核精度一致) ---
        if self.smooth_kernel is not None:
            with torch.no_grad():
                # 显式转换 kernel 精度
                kernel = self.smooth_kernel.to(p_score) 
                target_uns = target.unsqueeze(1)
                target_smooth = F.conv1d(target_uns, kernel, padding=self.padding).squeeze(1)
                target = torch.clamp(target_smooth, 0, 1)

        # --- 4. 【核心修复】数值安全钳制 (防止 NaN) ---
        # FP16 下 epsilon 不能太小，建议 1e-4 或 1e-5
        p_score = torch.clamp(p_score, 1e-5, 1.0 - 1e-5)

        # --- 5. 计算 Loss ---
        loss = F.binary_cross_entropy(p_score, target, reduction='none')

        # --- 6. 加权处理 ---
        pos_weight_t = torch.tensor(self.pos_weight, device=p_score.device, dtype=p_score.dtype)
        one_t = torch.tensor(1.0, device=p_score.device, dtype=p_score.dtype)
        pixel_weights = torch.where(target > 0.01, pos_weight_t, one_t)
        
        # 避免除以 0
        denom = mask.sum() + 1e-8
        loss = (loss * pixel_weights * mask).sum() / denom
        
        return loss