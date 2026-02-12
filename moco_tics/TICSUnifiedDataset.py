import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import re
import logging
import os

from transformers import XLMRobertaTokenizer
import torch

class TICSUnifiedDataset(Dataset):
    def __init__(self, pt_files, xlmr_model_path="/mnt/conda_data/facebook/xlm-roberta-base", target_sr=16000, min_frames=10, max_k=6, max_text_len=128):
        """
        xlmr_model_path: 传入你本地的 facebook/xlm-roberta-base 路径
        """
        self.files = pt_files
        self.target_sr = target_sr
        self.min_frames = min_frames
        self.max_k = max_k
        self.max_text_len = max_text_len
        
        # 实时初始化分词器 (放在 init 里只初始化一次)
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(xlmr_model_path)

    def _resample_feats(self, feats, target_len):
        # ... 保持你之前的代码逻辑不变 ...
        curr_len = feats.size(0)
        if curr_len == target_len: return feats
        if curr_len > target_len: return feats[:target_len, :]
        padding_len = target_len - curr_len
        padding = feats[-1:, :].repeat(padding_len, 1)
        return torch.cat([feats, padding], dim=0)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            data = torch.load(self.files[idx], map_location='cpu')
            
            # 1. 基础数据提取
            wav_path = data['wav_path']
            text = data.get('text', "")
            target_k = float(data.get('segment_count_nltk', 0))
            
            # 2. 严格过滤 (k=0 或超过上限则抛弃)
            if target_k <= 0 or target_k > self.max_k:
                return None

            wav_path = data.get('wav_path', "")
            if not os.path.exists(wav_path):
                # 使用红色表示严重错误
                print(f"\033[91m[MISSING WAV]\033[0m {wav_path} )")
                return None
            # 3. 音频加载与采样率对齐
            waveform, sr = torchaudio.load(wav_path)
            if sr != self.target_sr:
                waveform = torchaudio.transforms.Resample(sr, self.target_sr)(waveform)
            waveform = waveform.mean(0)

            # 4. 边界真值与情感真值对齐
            y_boundary = data['boundary'].float()
            T = y_boundary.size(0)
            if T < self.min_frames: return None
            
            emo_feats = self._resample_feats(data['emotion_feats'].float(), T)

            # 5. [核心优化] 实时生成文本导师所需的 Input IDs
            text_encoded = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.max_text_len,
                return_tensors="pt"
            )

            return {
                "wav": waveform,
                "y_boundary": y_boundary,
                "target_k": target_k,
                "target_emo": emo_feats,
                "text_input_ids": text_encoded['input_ids'].squeeze(0), # [L]
                "text_mask": text_encoded['attention_mask'].squeeze(0), # [L]
                "text": text,
                "mask": torch.ones(T)
            }

        except Exception as e:
            logging.warning(f"Error loading {self.files[idx]}: {e}")
            return None

def collate_fn_unified(batch):
    """
    处理不同长度样本的补齐 (Padding)，确保所有张量在 Batch 维度上对齐。
    """
    # 1. 过滤掉 Dataset 可能返回的 None (因错误或过滤逻辑产生的空样本)
    batch = [b for b in batch if b is not None]
    if not batch: 
        return {}

    # 2. 物理波形补齐 [B, Max_Wav_T]
    # 对齐原始音频采样，用于 Hubert 输入
    wav = pad_sequence([b['wav'] for b in batch], batch_first=True, padding_value=0)
    
    # 3. 物理边界补齐 [B, Max_T]
    # 0 代表非边界，1 代表词边界。补齐部分设为 0 是正确的。
    y_boundary = pad_sequence([b['y_boundary'] for b in batch], batch_first=True, padding_value=0)
    
    # 4. 情感特征导师补齐 [B, Max_T, 1024]
    # 用于计算 Emotion Consistency Loss
    target_emo = pad_sequence([b['target_emo'] for b in batch], batch_first=True, padding_value=0)
    
    # 5. 物理层 Mask 补齐 [B, Max_T]
    # 1 代表有效帧，0 代表 Padding 帧
    mask = pad_sequence([b['mask'] for b in batch], batch_first=True, padding_value=0)
    
    # 6. 文本导师输入补齐 (固定长度处理)
    # 因为在 Dataset 中我们用了 padding='max_length'，所以这里可以直接 stack
    # 如果 Dataset 中没设 max_length，则也需要用 pad_sequence
    text_input_ids = torch.stack([b['text_input_ids'] for b in batch])
    text_mask = torch.stack([b['text_mask'] for b in batch])
    
    # 7. 标量数据转换
    # target_k 必须是 float32 以适配 MSE/SmoothL1 回归损失
    target_ks = torch.tensor([b['target_k'] for b in batch], dtype=torch.float32)
    
    # 8. 原始文本 (仅用于分析和日志，不进模型)
    texts = [b['text'] for b in batch]

    return {
        "wav": wav,                   # 原始音频
        "y_boundary": y_boundary,     # 词边界标签 (0/1)
        "target_k": target_ks,        # NLTK 句子数
        "target_emo": target_emo,     # 情感特征真值
        "text_input_ids": text_input_ids, # 文本导师 ID
        "text_mask": text_mask,       # 文本导师 Mask
        "mask": mask,                 # 物理帧有效 Mask
        "texts": texts
    }