import os
import glob
import torch
import torch.distributed as dist
import argparse
import datetime
import deepspeed
from torch.utils.data import DataLoader, DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm
# 导入你自己的模块
from moco_tics.TICSUnifiedDataset import TICSUnifiedDataset, collate_fn_unified
from moco_tics.TICSMoCo import TICS_MoCo
from moco_tics.TICSLossCriterion import TICSLossCriterion

import logging

# --- 在 main() 的开始部分初始化日志 ---
def setup_logging(log_dir, local_rank):
    log_file = os.path.join(log_dir, f"train_stage2_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # 只有主进程负责写入文件和打印
    if local_rank == 0:
        handlers = [
            logging.FileHandler(log_file),
            logging.StreamHandler() # 同时输出到控制台
        ]
    else:
        handlers = [logging.NullHandler()] # 其他进程保持沉默

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser(description="TICS Stage 2 Training")
    parser.add_argument('--pt_dir', type=str, required=True, help="预处理好的 .pt 目录")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--save_steps', type=int, default=5000, help="每多少步保存一次断点")
    parser.add_argument('--log_steps', type=int, default=10, help="每多少步打印一次日志")
    parser.add_argument('--ds_config', type=str, required=True, help="DeepSpeed JSON 配置文件")
    
    # 允许从命令行覆盖 DeepSpeed 内部参数
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()

def main():
    args = get_args()

    # --- 1. 初始化分布式环境 ---
    deepspeed.init_distributed(timeout=datetime.timedelta(seconds=7200))
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # --- 2. 目录初始化 ---
    shm_checkpoint_dir = "/dev/shm/tics_stage2/checkpoints" # 极速缓存
    final_checkpoint_dir = "/data/tics_training_stage2/final_checkpoints" # 硬盘持久化
    log_dir = "/data/tics_training_stage2/logs"
    
    if local_rank == 0:
        os.makedirs(shm_checkpoint_dir, exist_ok=True)
        os.makedirs(final_checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
    else:
        writer = None

    # --- 3. 模型、损失函数与数据集 ---
    model = TICS_MoCo(
        backbone_path="/mnt/conda_data/facebook/hubert-base-ls960",
        xlmr_path="/mnt/conda_data/facebook/xlm-roberta-base",
        large_hubert_path="/mnt/conda_data/facebook/hubert-large-ls960-ft",
        is_stage2=True
    )

    criterion = TICSLossCriterion(
        pos_weight=15.0, # 针对边界稀疏性
        alpha=1.0, beta=2.0, gamma=1.0, delta=0.8 # 各 Loss 权重
    ).to(device)

    pt_files = glob.glob(os.path.join(args.pt_dir, "**/*.pt"), recursive=True)
    dataset = TICSUnifiedDataset(pt_files, xlmr_model_path="/mnt/conda_data/facebook/xlm-roberta-base")
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        collate_fn=collate_fn_unified, num_workers=4, pin_memory=True
    )

    # --- 4. DeepSpeed 初始化 ---
    # 这里会自动根据 ds_config 创建 optimizer 和 lr_scheduler
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        config=args.ds_config
    )

    # --- 5. 断点续练逻辑 (Auto Resume) ---
    # DeepSpeed 会自动查找最新的 checkpoint
    load_path, client_sd = model_engine.load_checkpoint(shm_checkpoint_dir)
    start_epoch = 0
    start_step = 0
    if load_path is not None:
        start_epoch = client_sd['epoch']
        start_step = client_sd['step']
        if local_rank == 0:
            print(f"✅ 已恢复断点: Epoch {start_epoch}, Step {start_step}")

    # --- 6. 训练循环 ---
    model_engine.train()
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), disable=(local_rank != 0))
        
        for step, batch in pbar:
            # 跳过已经训练过的步数 (断点续练时使用)
            if epoch == start_epoch and step < start_step:
                continue

            # 将 batch 移动到 GPU
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            # 前向传播
            model_output = model_engine(
                wav=batch['wav'],
                text_input_ids=batch['text_input_ids'],
                text_mask=batch['text_mask']
            )

            # 计算多维度 Loss
            loss, loss_dict = criterion(model_output, batch)
            # --- DEBUG 打印 ---
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n[RANK {local_rank} FATAL] Loss is NaN!")
                print(f"Details: {loss_dict}")
                continue # 跳过这一步，不要backward，防止污染模型权重
            # 反向传播 (DeepSpeed 自动处理 scale_loss)
            model_engine.backward(loss)
            model_engine.step()

            # --- 7. 日志与监控 ---
            if local_rank == 0 and step % args.log_steps == 0:
                # 记录 Tensorboard
                for name, val in loss_dict.items():
                    writer.add_scalar(f"Loss/{name}", val, epoch * len(dataloader) + step)
                
                # 特色监控：预测 K 值 vs 真实 K 值
                avg_pred_k = model_output['pred_k'].mean().item()
                avg_target_k = batch['target_k'].mean().item()
                writer.add_scalar("Monitor/Pred_K", avg_pred_k, epoch * len(dataloader) + step)
                writer.add_scalar("Monitor/Target_K", avg_target_k, epoch * len(dataloader) + step)

                # 3. 【重点修改】详细化的进度条打印
                # 这里我们直接从 loss_dict 中取值，并保留 3 位小数
                # B: 边界损失, D: 蒸馏损失, C: 数量预测损失
                pbar.set_description(
                    f"Epoch {epoch} | "
                    f"L:{loss.item():.3f} | "
                    f"B:{loss_dict['bnd']:.3f} | "
                    f"D:{loss_dict['dist']:.3f} | "
                    f"C:{loss_dict['count']:.3f} | "
                    f"K:{avg_pred_k:.1f}/{avg_target_k:.1f}"
                )
            # --- 8. 按步保存 (保存到 /dev/shm) ---
            if step > 0 and step % args.save_steps == 0:
                client_sd = {'epoch': epoch, 'step': step}
                model_engine.save_checkpoint(shm_checkpoint_dir, client_state=client_sd)

        # --- 9. Epoch 结束保存 (持久化到硬盘) ---
        dist.barrier()
        if local_rank == 0:
            print(f"💾 Epoch {epoch} 完成，正在持久化保存...")
            # 将内存中的最新模型复制到硬盘持久化存储
            model_engine.save_checkpoint(final_checkpoint_dir, tag=f"epoch_{epoch}", client_state={'epoch': epoch, 'step': 0})

    if local_rank == 0:
        writer.close()
        print("🎉 训练完成!")

if __name__ == "__main__":
    main()