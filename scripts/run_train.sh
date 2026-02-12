# 1. 环境变量设置
export PYTHONPATH=$PYTHONPATH:$(pwd)
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_TIMEOUT=7200000

# 2. 执行 DeepSpeed
# 注意：我们用满 8 张卡 (--num_gpus 8) 来分摊显存压力
deepspeed --num_gpus 8 \
    --master_port 29500 \
    scripts/train_end2end.py \
    --pt_dir /data/NaturalVoices/Stage2_Features \
    --batch_size 2 \
    --epochs 20 \
    --save_steps 5000 \
    --ds_config scripts/ds_config_t4.json