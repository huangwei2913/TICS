import torch
import torchaudio
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import XLMRobertaTokenizer
from typing import List, Tuple, Dict
import pandas as pd
import os
import json

class BoundaryLabelGenerator:
    def __init__(self, fps=50):
        self.fps = fps

    def generate(self, json_path: str, target_frames: int) -> torch.Tensor:
        """
        基于给定的帧数生成 0/1 序列
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            y_true = torch.zeros(target_frames, dtype=torch.float32)
            words = data.get('words', data.get('word_segments', []))
            
            for word in words:
                # 提取词边界结束时间
                end_time = word['end']
                # 转换到帧索引: frame = time * 50
                frame_idx = int(round(end_time * self.fps))
                
                # 严格边界检查：防止计算出的索引超出特征长度
                if frame_idx < target_frames:
                    y_true[frame_idx] = 1.0
                elif frame_idx == target_frames: # 容错处理
                    y_true[target_frames - 1] = 1.0
                    
            return y_true
        except Exception as e:
            # 如果 JSON 损坏，返回全 0，防止训练中断
            return torch.zeros(target_frames, dtype=torch.float32)

class TICSDataset(Dataset):
    def __init__(self, csv_path: str, sample_rate: int = 16000, xlmr_path="/mnt/facebook/xlm-roberta-large", stage=1):
        """
        csv_path: 第一列 wav 路径, 第二列 json 路径
        """
        self.stage = stage
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(xlmr_path)
        df = pd.read_csv(csv_path, header=None) # 假设没有表头，或者您指定 header=0
        self.audio_files = df.iloc[:, 0].tolist()
        self.json_files = df.iloc[:, 1].tolist()
        
        self.sample_rate = sample_rate
        self.label_gen = BoundaryLabelGenerator(fps=50)

    def __len__(self):
        return len(self.audio_files)

    def _apply_augmentation(self, waveform: torch.Tensor, is_view2: bool) -> torch.Tensor:
        # View 1: 原始/简单增益
        # View 2: 时间反转 + 增益
        # 注意：此处只做非变速增强，确保时间轴一致
        aug_wav = waveform.clone()
        if is_view2:
            aug_wav = torch.flip(aug_wav, dims=[1])
            
        gain = random.uniform(0.8, 1.2)
        aug_wav = aug_wav * gain
        return torch.clamp(aug_wav, -1.0, 1.0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        audio_path = self.audio_files[idx]
        json_path = self.json_files[idx]


        with open(json_path, 'r') as f:
            meta = json.load(f)

        # 1. 加载音频
        waveform, sr = torchaudio.load(audio_path)
        
        # 强制单声道
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        # 2. 计算 HuBERT 预期的特征帧数
        # HuBERT Stride 是 320 (20ms)，所以 T = L // 320
        target_T = waveform.shape[1] // 320
        
        if target_T == 0: # 过滤极短音频
            return self.__getitem__((idx + 1) % len(self))

        # 3. 生成 Y_true
        y_true = self.label_gen.generate(json_path, target_T)

        # 4. 生成两个增强视图
        view1 = self._apply_augmentation(waveform, is_view2=False)
        view2 = self._apply_augmentation(waveform, is_view2=True)

        if self.stage == 2:
            # 4. 文本 Token 处理 (Stage 2 专属)
            text = meta['text']
            # 对文本进行编码
            encoded_text = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=128,  # 根据你的平均句长调整
                return_tensors='pt'
            )
            
            return {
                "view1": view1, # 语音张量
                "y_true": y_true, # 边界标签
                "text_ids": encoded_text['input_ids'].squeeze(0),
                "text_mask": encoded_text['attention_mask'].squeeze(0)
            }


        return {
            "view1": view1.squeeze(0),
            "view2": view2.squeeze(0),
            "y_true": y_true
        }


def tics_collate_fn(batch):
    """
    兼容 Stage I 和 Stage II 的动态 Padding 函数
    """
    # 1. 基础项提取 (所有阶段共有)
    view1_list = [item['view1'] for item in batch]
    y_true_list = [item['y_true'] for item in batch]
    
    # 对音频和边界标签进行 Padding
    padded_view1 = pad_sequence(view1_list, batch_first=True, padding_value=0.0)
    padded_y_true = pad_sequence(y_true_list, batch_first=True, padding_value=0.0)
    
    # 生成音频掩码 y_mask (用于 Boundary Loss 排除 padding 部分)
    # y_true 形状为 (B, T)，y_mask 在有效长度为 1，padding 为 0
    lengths = [len(y) for y in y_true_list]
    max_len = max(lengths)
    y_mask = torch.zeros((len(batch), max_len), dtype=torch.float32)
    for i, l in enumerate(lengths):
        y_mask[i, :l] = 1.0

    # 构造基础返回字典
    output = {
        "view1": padded_view1,
        "y_true": padded_y_true,
        "y_mask": y_mask
    }

    # 2. Stage I 特有项：处理 view2 (对比视图)
    if 'view2' in batch[0]:
        view2_list = [item['view2'] for item in batch]
        output["view2"] = pad_sequence(view2_list, batch_first=True, padding_value=0.0)

    # 3. Stage II 特有项：处理文本 Token
    if 'text_ids' in batch[0]:
        text_ids_list = [item['text_ids'] for item in batch]
        text_mask_list = [item['text_mask'] for item in batch]
        
        # 文本通常在 Dataset 里已经固定了 max_length，但保险起见这里再做一次 pad
        output["text_ids"] = pad_sequence(text_ids_list, batch_first=True, padding_value=1) # XLM-R pad ID 通常是 1，请根据 tokenizer 确认
        output["text_mask"] = pad_sequence(text_mask_list, batch_first=True, padding_value=0.0)

    return output



def get_tics_dataloader(csv_path: str, batch_size: int, num_workers: int):
    dataset = TICSDataset(csv_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=tics_collate_fn,
        pin_memory=True
    )