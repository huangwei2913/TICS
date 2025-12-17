import argparse
import deepspeed
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from moco_tics.model import TICS_MoCo
from moco_tics.data_loader import TICSDataset, tics_collate_fn


##################
'''正在从本地加载 XLM-R 模型: /mnt/facebook/xlm-roberta-large ...
Some weights of the model checkpoint at /mnt/facebook/xlm-roberta-large were not used when initializing XLMRobertaModel: ['lm_head.dense.bias', 'lm_head.layer_norm.weight', 'lm_head.bias', 'lm_head.layer_norm.bias', 'lm_head.dense.weight']
- This IS expected if you are initializing XLMRobertaModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing XLMRobertaModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).

==============================
--- Tokenizer 特殊 Token 确认 ---
Padding Token: '<pad>'
Padding Token ID: 1
BOS Token ID: 0
EOS Token ID: 2
==============================

--- Token 序列测试 ---
Tokens: ['<s>', '▁他', '方', '今天', '给', '纯', '爱', '家', '是', '暂时', '加', '了很多', '条件', '。', '只要你', '对', '</s>']
IDs: [0, 47437, 2698, 7461, 4766, 71781, 7558, 1433, 354, 135030, 3490, 49120, 12093, 30, 172946, 1036, 2]

--- Padding 行为验证 ---
Batch IDs 矩阵:
tensor([[     0,      6,  10119,  27683,   1344,      2,      1,      1,      1],
        [     0,      6, 100013,   4528,  83944,  27683,   1344,  49125,      2]])

'''
#############


# --- 1. 语义蒸馏损失 (全局池化对齐) ---
class TICS_Stage2_Loss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_audio_proj, z_text, audio_mask, text_mask):
        """
        audio_mask: True 代表 Padding (Transformer 格式)
        text_mask: 1.0 代表有效 (HuggingFace 格式)
        """
        # 处理语音全局特征 (取有效位置平均)
        a_mask = (~audio_mask).unsqueeze(-1).float() 
        audio_global = (z_audio_proj * a_mask).sum(dim=1) / (a_mask.sum(dim=1) + 1e-6)

        # 处理文本全局特征 (取有效位置平均)
        t_mask = text_mask.unsqueeze(-1).float()
        text_global = (z_text * t_mask).sum(dim=1) / (t_mask.sum(dim=1) + 1e-6)

        # 计算 Cosine 相似度损失
        cos_sim = F.cosine_similarity(audio_global, text_global, dim=-1)
        loss_distill = (1.0 - cos_sim).mean()
        
        return loss_distill

# --- 2. 带有掩码的边界损失 (物理对齐) ---
class MaskedBoundaryLoss(nn.Module):
    def __init__(self, pos_weight=15.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, p_score, y_true, mask):
        # 确保维度对齐
        min_t = min(p_score.size(1), y_true.size(1))
        p_score = p_score[:, :min_t]
        y_true = y_true[:, :min_t].float()
        mask = mask[:, :min_t].float()

        # 加权 BCE
        loss = F.binary_cross_entropy(p_score, y_true, reduction='none')
        weight = 1.0 + y_true * (self.pos_weight - 1.0)
        
        masked_loss = (loss * weight * mask).sum() / (mask.sum() + 1e-6)
        return masked_loss

def parse_args():
    parser = argparse.ArgumentParser(description="TICS Stage II - Semantic Distillation")
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--stage1_checkpoint', type=str, required=True)
    parser.add_argument('--xlmr_path', type=str, default="/mnt/facebook/xlm-roberta-large")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lambda_sup', type=float, default=1.0)
    parser.add_argument('--lambda_distill', type=float, default=2.0)
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. 模型初始化
    TEACHER_CONFIG = {
        'input_dim': 768,
        'segment_dim': 1024,
        'num_layers': 12,
        'dropout': 0.1
    }
    
    # 开启 is_stage2 开关，加载 XLM-R 老师
    model = TICS_MoCo(
        backbone_path="/mnt/facebook/hubert-base-ls960",
        teacher_config=TEACHER_CONFIG,
        xlmr_path=args.xlmr_path,
        is_stage2=True
    )

    # 2. 加载 Stage 1 训练好的切分能力
    print(f"Loading Stage I weights: {args.stage1_checkpoint}")
    state_dict = torch.load(args.stage1_checkpoint, map_location='cpu')
    # strict=False 因为 stage2 新增了投影层和老师权重
    model.load_state_dict(state_dict, strict=False)

    # 3. 数据集准备
    train_dataset = TICSDataset(csv_path=args.csv_path, xlmr_path=args.xlmr_path, stage=2)
    
    # DeepSpeed 初始化
    # 它会自动根据 ds_config 和 training_data 创建适当的 DataLoader
    model_engine, optimizer, train_loader, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        collate_fn=tics_collate_fn,
    )

    # 4. 定义损失函数
    distill_criterion = TICS_Stage2_Loss().to(model_engine.device)
    boundary_criterion = MaskedBoundaryLoss(pos_weight=15.0).to(model_engine.device)

    model_engine.train()
    for epoch in range(100):
        for step, batch in enumerate(train_loader):
            # 将数据送入 GPU (Stage 2 只需要单视图 view1)
            wav = batch["view1"].to(model_engine.device).half()
            y_true = batch["y_true"].to(model_engine.device)
            y_mask = batch["y_mask"].to(model_engine.device)
            text_ids = batch["text_ids"].to(model_engine.device)
            text_mask = batch["text_mask"].to(model_engine.device)

            # --- 前向传播 (使用 Stage 2 特有的语义蒸馏路径) ---
            # 通过 .module 访问原始模型中的自定义方法
            outputs = model_engine.module.forward_stage2(wav, text_ids, text_mask)

            # --- 计算 Loss ---
            # A. 语义蒸馏: 迫使语音片段 z 靠近 XLM-R 的文本语义
            loss_distill = distill_criterion(
                outputs["z_audio_proj"], 
                outputs["z_text"], 
                outputs["audio_mask"], 
                outputs["text_mask"]
            )

            # B. 边界监督: 确保物理断句依然对齐标注
            loss_sup = boundary_criterion(outputs["p_score"], y_true, y_mask)

            total_loss = args.lambda_distill * loss_distill + args.lambda_sup * loss_sup

            # --- 反向传播 ---
            model_engine.backward(total_loss)
            model_engine.step()

            if step % 10 == 0 and args.local_rank <= 0:
                print(f"Epoch {epoch} | Step {step} | Total: {total_loss.item():.4f} | "
                      f"Distill: {loss_distill.item():.4f} | Sup: {loss_sup.item():.4f}")

        # 保存 Epoch 检查点
        if args.local_rank <= 0:
            model_engine.save_checkpoint(save_dir="checkpoints_stage2", tag=f"epoch_{epoch}")



####
'''deepspeed --num_gpus=2 train_stage2.py \
    --deepspeed \
    --deepspeed_config ds_config_stage2.json \
    --csv_path ./data/emilia_train.csv \
    --stage1_checkpoint ./checkpoints_stage1/best_model.pt \
    --xlmr_path /mnt/facebook/xlm-roberta-large \
    --batch_size 8 \
    --lambda_distill 2.0 \
    --lambda_sup 1.0
'''
####



if __name__ == "__main__":
    main()