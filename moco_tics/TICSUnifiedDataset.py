import torch
import torchaudio
import pandas as pd
import os
import logging
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import XLMRobertaTokenizer

class TICSUnifiedDataset(Dataset):
    def __init__(self, csv_path, xlmr_model_path="/mnt/conda_data/facebook/xlm-roberta-base", 
                 target_sr=16000, min_frames=10, max_text_len=128, fps=50):
        """
        csv_path: 你的 tics_train_manifest.csv 路径
        fps: 特征每秒帧数 (Hubert 默认为 50, 即 20ms 一帧)
        """
        self.df = pd.read_csv(csv_path)
        self.target_sr = target_sr
        self.min_frames = min_frames
        self.max_text_len = max_text_len
        self.fps = fps
        
        # 实时初始化分词器
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(xlmr_model_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            
            # 1. 基础数据提取
            wav_path = row['wav_path']
            text = str(row['text']) if pd.notna(row['text']) else ""
            target_k = float(row['target_k'])
            
            # 2. 检查音频文件
            if not os.path.exists(wav_path):
                print(f"\033[91m[MISSING WAV]\033[0m {wav_path}")
                return None

            # 3. 音频加载与采样率对齐
            waveform, sr = torchaudio.load(wav_path)
            if sr != self.target_sr:
                waveform = torchaudio.transforms.Resample(sr, self.target_sr)(waveform)
            waveform = waveform.mean(0) # 转为单声道 [T]

            # 4. 物理边界 (y_boundary) 实时构建
            # 计算总帧数 (对应模型输出的序列长度)
            duration = waveform.shape[0] / self.target_sr
            num_frames = int(duration * self.fps)
            
            if num_frames < self.min_frames:
                return None
                
            y_boundary = torch.zeros(num_frames)
            
            # 解析 CSV 里的时间戳字符串 "0.42,1.66,2.91"
            if pd.notna(row['boundary']):
                time_stamps = [float(t) for t in str(row['boundary']).split(',')]
                for t in time_stamps:
                    frame_idx = int(t * self.fps)
                    if frame_idx < num_frames:
                        y_boundary[frame_idx] = 1.0

            # 5. 文本 Tokenize
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
                "text_input_ids": text_encoded['input_ids'].squeeze(0),
                "text_mask": text_encoded['attention_mask'].squeeze(0),
                "text": text,
                "mask": torch.ones(num_frames)
            }

        except Exception as e:
            logging.warning(f"Error processing index {idx}: {e}")
            return None

def collate_fn_unified(batch):
    """
    处理不同长度样本的补齐 (Padding)
    """
    batch = [b for b in batch if b is not None]
    if not batch: return {}

    # 波形补齐 (用于 Hubert 输入)
    wav = pad_sequence([b['wav'] for b in batch], batch_first=True, padding_value=0)
    
    # 物理边界补齐
    y_boundary = pad_sequence([b['y_boundary'] for b in batch], batch_first=True, padding_value=0)
    
    # Mask 补齐
    mask = pad_sequence([b['mask'] for b in batch], batch_first=True, padding_value=0)
    
    # 文本导师输入 (固定长度)
    text_input_ids = torch.stack([b['text_input_ids'] for b in batch])
    text_mask = torch.stack([b['text_mask'] for b in batch])
    
    # 标量数据转换
    target_ks = torch.tensor([b['target_k'] for b in batch], dtype=torch.float32)
    
    texts = [b['text'] for b in batch]

    return {
        "wav": wav,
        "y_boundary": y_boundary,
        "target_k": target_ks,
        "text_input_ids": text_input_ids,
        "text_mask": text_mask,
        "mask": mask,
        "texts": texts
    }