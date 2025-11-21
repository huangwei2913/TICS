# test/test_backbone.py

import sys
import os
import torch

# --- 【关键修改部分：添加项目根目录到路径】 ---
# 1. 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. 找到项目根目录 (即 TICS 目录)
project_root = os.path.join(current_dir, '..')
# 3. 将项目根目录添加到 Python 模块搜索路径
sys.path.append(project_root)
# --------------------------------------------------

# 现在就可以正确导入同级目录 moco_tics 下的模块了
from moco_tics.backbone import FrozenHubertBackbone
from transformers import HubertModel # 确保这个也已经安装

def test():
    # 创建一个伪造的音频 Batch (Batch=2, Time=16000*2s)
    # 注意：HubertModel.from_pretrained 默认期望输入是 (Batch, Samples)
    fake_wav = torch.randn(2, 32000) 
    
    # 请确保您已经完成了 conda activate tics 并且安装了所有依赖
    print("Attempting to load HuBERT backbone...")
    
    # 调整 output_layer，例如用第 9 层
    backbone = FrozenHubertBackbone(output_layer=9)
    
    # 将输入数据移到 GPU 上
    if torch.cuda.is_available():
        fake_wav = fake_wav.cuda()
        backbone = backbone.cuda()

    # 运行前向传播
    features = backbone(fake_wav)
    
    print("--- Test Successful ---")
    print(f"Output device: {features.device}")
    # 预期输出: torch.Size([2, 100, 768]) 
    print(f"Output shape: {features.shape}") 

if __name__ == "__main__":
    test()