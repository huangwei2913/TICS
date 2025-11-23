import torch
import torchaudio
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple
import torch
import torchaudio
import pandas as pd # <-- 缺少这一行
import random
import os

# --- 辅助函数：读取 CSV 路径列表 ---
def load_audio_paths_from_csv(csv_path: str) -> List[str]:
    """
    从 Aishell-1 的元数据 CSV 中加载音频相对路径列表。
    """
    try:
        # 假设 CSV 格式为 'Audio:FILE,Text:LABEL'，我们只需要第一列
        df = pd.read_csv(csv_path, header=0, usecols=[0], names=['AudioPath'])
        return df['AudioPath'].tolist()
    except Exception as e:
        print(f"Error reading CSV at {csv_path}: {e}")
        raise

# --- 1. 数据增强和变换函数 ---

def apply_audio_augmentation(waveform: torch.Tensor, is_view2: bool) -> torch.Tensor:
    """
    对音频应用不同的增强策略，用于生成两个视图。
    
    Args:
        waveform: (1, L) 原始音频波形张量。
        is_view2: 是否为 View 2 (应用时间反转)。
    
    Returns:
        增强后的波形。
    """
    
    # View 1: 标准增强 (例如，随机增益和噪声，这里为简化只做随机增益)
    if not is_view2:
        # 随机增益
        gain = random.uniform(0.8, 1.2)
        waveform = waveform * gain
        
    # View 2: 时间反转 (实现您要求的“反向音频”) + 随机增益
    else:
        # 理由：时间反转是核心变换，测试模型的时序不变性
        waveform = torch.flip(waveform, dims=[1])
        
        # 应用随机增益 (保持与 View 1 类似的幅度变化)
        gain = random.uniform(0.8, 1.2)
        waveform = waveform * gain
        
    # 裁剪到 [-1, 1]
    return torch.clamp(waveform, -1.0, 1.0)


# --- 2. 数据集类 ---

# --- 2. 数据集类 (已更新为 CSV 解析) ---
class TICSDataset(Dataset):
    """
    TICS 自监督训练数据集，从 CSV 加载路径并返回同一音频的两个增强视图。
    """
    def __init__(self, csv_path: str, data_root: str, sample_rate: int = 16000):
        # 加载所有相对路径
        self.relative_audio_paths = load_audio_paths_from_csv(csv_path)
        # 确保 data_root 是绝对路径，并且结尾没有斜杠
        self.data_root = os.path.abspath(data_root)
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.relative_audio_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        
        relative_path = self.relative_audio_paths[idx]
        
        # 核心：组合绝对路径，例如 /mnt/speech_asr_aishell_trainsets/wav/train/...
        full_path = os.path.join(self.data_root, relative_path)
        
        try:
            waveform, sr = torchaudio.load(full_path)
        except Exception as e:
            print(f"Skipping corrupt or unreadable file: {full_path}. Error: {e}")
            # 返回一个空占位符，由 collate_fn 过滤，或直接抛出错误
            raise IndexError(f"Error loading file: {full_path}")
        
        # 确保采样率一致 (如果需要)
        if sr != self.sample_rate:
             resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
             waveform = resampler(waveform)

        # 确保是单声道 (1, L)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # View 1: 正向音频 + 标准增强
        view1_wav = apply_audio_augmentation(waveform, is_view2=False)
        
        # View 2: 反向音频 + 标准增强 (时间反转)
        view2_wav = apply_audio_augmentation(waveform, is_view2=True)
        
        return view1_wav.squeeze(0), view2_wav.squeeze(0) # 变为 (L,) 方便 collate

# --- 3. 批次整理函数 (Collate Function) ---

def tics_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将批次中的可变长音频波形填充到批次中的最大长度。
    
    Args:
        batch: 一个包含 (View1_wav, View2_wav) 元组的列表。
        
    Returns:
        (padded_view1, padded_view2) 两个填充后的张量 (B, L_max)。
    """
    # 将 View 1 和 View 2 分离
    view1_list = [item[0] for item in batch]
    view2_list = [item[1] for item in batch]
    
    # 理由： pad_sequence 可以高效地处理可变长度并进行零填充
    
    # 填充 View 1
    # batch_first=True -> (B, L_max)
    padded_view1 = pad_sequence(view1_list, batch_first=True, padding_value=0.0)
    
    # 填充 View 2
    padded_view2 = pad_sequence(view2_list, batch_first=True, padding_value=0.0)

    # Note: 填充后的波形需要传递给 TICS_MoCo.forward
    # TICS_MoCo 内部的 FrozenHuBERT Backbone 会自行处理 padding/masking
    
    return padded_view1, padded_view2


# --- 示例 DataLoader 构造 ---
def get_tics_dataloader(audio_paths: List[str], batch_size: int, num_workers: int):
    """
    用于在 scripts/train.py 中实例化 DataLoader
    """
    dataset = TICSDataset(audio_paths)
    
    # 理由：使用 tics_collate_fn 来确保变长音频的正确批次化
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=tics_collate_fn,
        pin_memory=True
    )
    return dataloader