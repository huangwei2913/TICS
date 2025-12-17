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
from moco_tics.modules import FrozenHubertBackbone
from transformers import HubertModel # 确保这个也已经安装

def test():
    # 伪造一个 Batch=2、长度=2s(假设 16kHz) 的音频
    fake_wav = torch.randn(2, 32000)  # (batch, time_samples)

    print("Attempting to load HuBERT backbone...")

    # 实例化：可以传入自定义 hubert 路径，也可以用默认
    backbone = FrozenHubertBackbone(
        model_path="/mnt/facebook/hubert-base-ls960"
    )

    # 如果有 GPU，就搬到 GPU
    if torch.cuda.is_available():
        fake_wav = fake_wav.cuda()
        backbone = backbone.cuda()

    # 举例：提取 embedding(0)、第 6 层和第 12 层
    layers_to_extract = [0, 6, 12]

    # 前向传播
    features_list = backbone(fake_wav, layers_to_extract=layers_to_extract)

    print("--- Test Successful ---")
    for idx, feat in zip(layers_to_extract, features_list):
        print(f"Layer {idx} -> shape: {feat.shape}, device: {feat.device}")
    # 比如预期: torch.Size([2, T, 768])，T 大约是帧数
    
if __name__ == "__main__":
    test()