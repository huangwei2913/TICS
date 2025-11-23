import torch
import torch.nn.functional as F

# --- 1. 核心 NCE 损失函数 ---

def info_nce_loss(q: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor, temp: float) -> torch.Tensor:
    """
    计算 InfoNCE 损失。
    Args:
        q: (N, D) - Query 向量
        pos: (N, D) - Positive 向量
        neg: (N, K, D) - Negative 向量
    """
    # 1. 整合所有样本: (N, 1+K, D)
    all_samples = torch.cat([pos.unsqueeze(1), neg], dim=1)
    
    # 2. 计算相似度 (Query vs All)
    # q: (N, 1, D)
    # all_samples: (N, 1+K, D)
    # logits: (N, 1+K)
    logits = torch.einsum("nd, nkd -> nk", q.unsqueeze(1), all_samples) / temp
    
    # 3. 目标标签: 正样本在索引 0
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device) 
    
    # 4. 计算交叉熵
    loss = F.cross_entropy(logits, labels)
    return loss

# --- 2. 对称性损失 L_SYM (Transformational Invariance) ---

def compute_sym_loss(q1, k2, q2, k1, mask1, mask2, temp, K_neg) -> torch.Tensor:
    """
    计算对称性损失: L(Q1, K2) + L(Q2, K1) / 2
    使用批次内所有有效的 K 嵌入作为负样本池。
    """
    
    # 理由 1: 确保所有嵌入处于单位球上
    q1 = F.normalize(q1, dim=-1)
    k2 = F.normalize(k2, dim=-1)
    q2 = F.normalize(q2, dim=-1)
    k1 = F.normalize(k1, dim=-1)
    
    # 理由 2: 提取所有有效的 Key 嵌入作为负样本池
    # K1 和 K2 (动量教师的输出) 应该是最稳定的目标。
    # T, B, D -> (T*B, D)
    k1_flat = k1.flatten(end_dim=1)
    k2_flat = k2.flatten(end_dim=1)
    
    # 提取非填充 (有效) 的 K 嵌入
    # mask (B, T) -> (T, B) -> (T*B)
    valid_k1 = k1_flat[~mask1.transpose(0, 1).flatten()]
    valid_k2 = k2_flat[~mask2.transpose(0, 1).flatten()]
    
    # 整合负样本池 (包含 View 1 和 View 2 的所有有效 Key 嵌入)
    # Note: 这里的负样本池包含了正样本 (但 NCE 损失可以处理这个问题)
    negative_pool = torch.cat([valid_k1, valid_k2], dim=0) # (N_total, D)
    
    # 理由 3: 从 Q1 对 K2 的损失 (L_SYM_A)
    # 锚点 Q1 必须与对应的 K2 匹配
    
    # 简化策略: 我们只在**共同长度**上对齐 Q 和 K。
    # 由于 View1 和 View2 的分段数可能不同，我们只取最短公共长度 L_min
    L_min = min(q1.shape[0], k2.shape[0])
    
    # 提取对齐的 Query 和 Positive
    q1_aligned = q1[:L_min].flatten(end_dim=1)
    k2_aligned = k2[:L_min].flatten(end_dim=1)
    
    # 过滤掉填充位置的 Query 和 Positive (以防 L_min 仍包含填充)
    active_mask = ~mask1.transpose(0, 1)[:L_min].flatten()
    q1_active = q1_aligned[active_mask]
    k2_active = k2_aligned[active_mask]
    
    # 构造负样本张量 (N_active, K_neg, D)
    # 从负样本池中为每个活跃的 Q 随机采样 K_neg 个负样本
    
    # 这里使用一个简化的负采样方法:
    indices = torch.randint(0, negative_pool.shape[0], (q1_active.shape[0], K_neg), device=q1.device)
    negatives_a = negative_pool[indices]
    
    # 计算 L_SYM_A
    loss_sym_a = info_nce_loss(q1_active, k2_active, negatives_a, temp)
    
    # Q2 对 K1 的损失 (L_SYM_B)
    L_min_b = min(q2.shape[0], k1.shape[0])
    
    q2_aligned = q2[:L_min_b].flatten(end_dim=1)
    k1_aligned = k1[:L_min_b].flatten(end_dim=1)
    active_mask_b = ~mask2.transpose(0, 1)[:L_min_b].flatten()
    q2_active = q2_aligned[active_mask_b]
    k1_active = k1_aligned[active_mask_b]
    
    indices_b = torch.randint(0, negative_pool.shape[0], (q2_active.shape[0], K_neg), device=q2.device)
    negatives_b = negative_pool[indices_b]
    
    loss_sym_b = info_nce_loss(q2_active, k1_active, negatives_b, temp)
    
    return (loss_sym_a + loss_sym_b) / 2.0

# --- 3. 预测性损失 L_CPC (Temporal Predictability) ---

def compute_cpc_loss(q1, k1, mask1, temp, max_step=12, K_neg=100) -> torch.Tensor:
    """
    计算预测性损失: Q1(t) 预测 K1(t+k)
    """
    T, B, D = q1.shape
    device = q1.device
    
    # 理由 1: 预测性损失通常使用 Prediction Head 的输出 (q1)
    q1 = F.normalize(q1, dim=-1) # Q 是 projected Z_t
    k1 = F.normalize(k1, dim=-1) # K 是 target Z_t+k
    
    # 理由 2: 随机选择预测步长 k
    k = torch.randint(1, max_step + 1, (1,)).item()
    
    # 锚点和目标索引
    anchor_indices = torch.arange(T - k, device=device) 
    target_indices = torch.arange(k, T, device=device)
    
    # 提取 Z_t (Query) 和 Z_t+k (Positive Target)
    q_t = q1[anchor_indices].flatten(end_dim=1)     # (N_anchors, D)
    k_t_plus_k = k1[target_indices].flatten(end_dim=1) # (N_anchors, D)
    
    # 理由 3: 提取 Mask
    mask_T_B = mask1.transpose(0, 1) # (T, B)
    anchor_mask_flat = mask_T_B[anchor_indices].flatten() # (N_anchors)
    
    # 提取有效的 Query 和 Positive
    q_t_active = q_t[~anchor_mask_flat]
    k_t_plus_k_active = k_t_plus_k[~anchor_mask_flat]
    
    if q_t_active.numel() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # 理由 4: 构造负样本 (Batch-wise Negatives)
    # 使用 K1 中所有有效的非目标片段作为负样本池
    k1_flat = k1.flatten(end_dim=1)
    negative_pool = k1_flat[~mask1.transpose(0, 1).flatten()]
    
    # 随机采样
    indices = torch.randint(0, negative_pool.shape[0], (q_t_active.shape[0], K_neg), device=device)
    negatives = negative_pool[indices]
    
    # 计算损失
    loss_cpc = info_nce_loss(q_t_active, k_t_plus_k_active, negatives, temp)
    
    return loss_cpc

# --- 4. 总损失函数 ---

def compute_tics_loss(outputs: dict, 
                      temperature: float = 0.1, 
                      lambda_cpc: float = 1.0, # L_CPC 权重
                      lambda_sym: float = 1.0, # L_SYM 权重
                      num_negatives: int = 256,
                      max_step: int = 12) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    计算 TICS 总损失: L_Total = lambda_sym * L_SYM + lambda_cpc * L_CPC
    """
    
    # L_SYM: 对称性损失
    loss_sym = compute_sym_loss(
        outputs['q1'], outputs['k2'], outputs['q2'], outputs['k1'],
        outputs['mask_q1'], outputs['mask_q2'], temperature, num_negatives
    )
    
    # L_CPC: 预测性损失
    loss_cpc = compute_cpc_loss(
        outputs['q1'], outputs['k1'], outputs['mask_q1'],
        temperature, max_step, num_negatives
    )
    
    # L_Total
    total_loss = lambda_sym * loss_sym + lambda_cpc * loss_cpc
    
    # 返回损失和日志
    loss_log = {
        "total_loss": total_loss.item(),
        "sym_loss": loss_sym.item(),
        "cpc_loss": loss_cpc.item()
    }
    
    return total_loss, loss_log