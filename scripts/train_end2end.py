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
os.environ["NCCL_BLOCKING_WAIT"] = "1" 
os.environ["NCCL_TIMEOUT"] = "1800" # 统一为 30 分钟
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

# 初始化全功能 Loss
    criterion = TICSLossCriterion(
        pos_weight=15.0, 
        alpha=1.0,   # Boundary
        beta=1.0,    # MoCo
        gamma=0.1,   # MSE (语义脑补)
        lambda_k=0.5 # Count (数量回归)
    ).to(device)

    #pt_files = glob.glob(os.path.join(args.pt_dir, "**/*.pt"), recursive=True)
    pt_files = args.pt_dir
    csv_path=pt_files
    dataset = TICSUnifiedDataset(csv_path, xlmr_model_path="/mnt/conda_data/facebook/xlm-roberta-base")
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
            loss_dict = criterion(model_output, batch, model_engine.module.queue)
            total_loss = loss_dict["loss"]
            # --- C. 反向传播与优化 ---
            if not (torch.isnan(total_loss) or torch.isinf(total_loss)):
                model_engine.backward(total_loss)
                model_engine.step()
                
                # --- D. 【核心修改】更新 MoCo 队列 ---
                # 必须在 step() 之后更新，且只需使用教师产出的 k_m
                with torch.no_grad():
                    model_engine.module._dequeue_and_enqueue(model_output["k_m"])
            else:
                if local_rank == 0: print(f"⚠️ Warning: NaN Loss at Step {step}")
                continue

            # --- 7. 日志与监控 ---
            if local_rank == 0 and step % args.log_steps == 0:
                # Tensorboard 记录
                for name, val in loss_dict.items():
                    if isinstance(val, torch.Tensor): val = val.item()
                    writer.add_scalar(f"Loss/{name}", val, epoch * len(dataloader) + step)
                
                # 计算 K 值偏差
                avg_pred_k = model_output['pred_k'].mean().item()
                avg_target_k = batch['target_k'].mean().item()

                # 精简后的进度条打印 (B:Boundary, M:MoCo, S:Semantic/MSE, C:Count)
                pbar.set_description(
                    f"Epoch {epoch} | "
                    f"Loss:{total_loss.item():.3f} | "
                    f"B:{loss_dict['loss_boundary']:.3f} | "
                    f"M:{loss_dict['loss_moco']:.3f} | "
                    f"S:{loss_dict['loss_mse']:.3f} | "
                    f"C:{loss_dict['loss_count']:.3f} | "
                    f"K_Acc:{avg_pred_k:.1f}/{avg_target_k:.1f}"
                )
            # --- 8. 按步保存 (保存到 /dev/shm) ---

            if step > 0 and step % args.save_steps == 0:
                # [屏障 1]: 确保所有卡都跑完了这一步的计算，准备好保存
                dist.barrier() 
                
                if local_rank == 0:
                    print(f"--- All ranks reached checkpoint trigger at step {step}. Starting save... ---")

                # 【关键点】: 不要放在 if local_rank == 0 里面！
                # DeepSpeed 的 save_checkpoint 内部会自动处理多卡的同步和分片写入。
                # 所有 Rank 必须一起调用它。
                client_sd = {'epoch': epoch, 'step': step}
                model_engine.save_checkpoint(shm_checkpoint_dir, client_state=client_sd)

                # [屏障 2]: 确保所有卡都写完了自己的分片，再一起往后走
                dist.barrier()
                
                if local_rank == 0:
                    print(f"--- Step {step} checkpoint saved successfully. Resuming training... ---")

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