# 1. 环境变量设置
export PYTHONPATH=$PYTHONPATH:$(pwd)
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=2,3,4,5  # <--- 核心修改：只看这四张空闲卡
# NCCL 超时设置：建议设为 3600 (1小时)，这是一个既能容忍保存延迟，又能及时发现死锁的黄金点
export NCCL_TIMEOUT=7200
# 开启这个变量，当发生超时时，日志会打印出到底是哪个 Rank 在哪一步卡住了
export NCCL_DEBUG=INFO 
export TORCH_DISTRIBUTED_DEBUG=DETAIL
# 强制每个操作完成后检查错误，而不是在那干等
export NCCL_ASYNC_ERROR_HANDLING=1
# --- 稳定性开关 ---
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1

# --- 超时设置 ---
# NCCL 超时 (通讯)
export NCCL_TIMEOUT=7200  # 2小时
# Gloo 超时 (屏障监控) - 你的报错是 monitoredBarrier，所以这个很重要！
export TORCH_DISTRIBUTED_DETAIL=DEBUG
# 打印更多调试信息
# 2. 执行 DeepSpeed
# 注意：我们用满 8 张卡 (--num_gpus 8) 来分摊显存压力
export CUDA_VISIBLE_DEVICES=2,3,4,5
export PYTHONPATH=$PYTHONPATH:$(pwd)
deepspeed \
    --include localhost:2,3,4,5 \
    --master_port 29500 \
    scripts/train_end2end.py \
    --pt_dir /data/Emilia/tics_train_manifest.csv \
    --batch_size 2 \
    --epochs 20 \
    --save_steps 5000 \
    --ds_config scripts/ds_config_t4.json