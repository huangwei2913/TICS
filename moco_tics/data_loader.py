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
from .TicsAugmentation import TicsAugmentation
import pandas as pd
import orjson
from tqdm import tqdm


class BoundaryLabelGenerator:
    def __init__(self, fps=50):
        self.fps = fps

    def generate(self, data: dict, target_frames: int) -> torch.Tensor:
        """
        不再读磁盘，直接处理传入的内存字典对象
        """
        try:
            # 此时 data 已经是 TICSDataset 传进来的字典了
            if data is None:
                return torch.zeros(target_frames, dtype=torch.float32)

            y_true = torch.zeros(target_frames, dtype=torch.float32)
            words = data.get('words', data.get('word_segments', []))
            
            for word in words:
                # 转换到帧索引: frame = time * 50
                frame_idx = int(round(word['end'] * self.fps))
                
                # 严格边界检查
                if frame_idx < target_frames:
                    y_true[frame_idx] = 1.0
                elif frame_idx == target_frames: 
                    y_true[target_frames - 1] = 1.0
                    
            return y_true
        except Exception as e:
            # 打印错误方便排查，但返回全 0 保证训练不崩溃
            print(f"Label generation error: {e}")
            return torch.zeros(target_frames, dtype=torch.float32)

class TICSDataset(Dataset):
    def __init__(self, csv_path: str, sample_rate: int = 16000, xlmr_path="/mnt/facebook/xlm-roberta-large", augmentor=None, stage=1):
        self.stage = stage
        self.sample_rate = sample_rate
        
        # 1. 更加健壮的 CSV 加载
        print(f"🔍 正在加载 CSV 文件: {csv_path}")
        # 如果你的 CSV 确实没有表头，用 header=None；如果有，用 header=0
        df = pd.read_csv(csv_path, header=None) 
        # 2. 核心：过滤掉非路径的无效行（比如表头文字）
        # 只有当第一列包含 '/' (路径特征) 且不为空时才保留
        valid_mask = df.iloc[:, 0].str.contains('/', na=False)
        df = df[valid_mask]
        
        self.audio_files = df.iloc[:, 0].tolist()
        self.json_files = df.iloc[:, 1].tolist() 
        
        self.cached_json = []
        print(f"🚀 内存充足，正在预加载 {len(self.json_files)} 条 JSON 特征...")
        
        # 预加载循环
        for json_p in tqdm(self.json_files, desc="Caching JSONs", unit="file"):
            try:
                with open(json_p, 'rb') as f:
                    # 使用 orjson 快速解析二进制流
                    # 存入列表后，__getitem__ 访问速度是 O(1) 且零磁盘 IO
                    self.cached_json.append(orjson.loads(f.read()))
            except Exception as e:
                print(f"Error loading {json_p}: {e}")
                # 为了索引对应，加载失败也占个位（或者在之前过滤掉不存在的文件）
                self.cached_json.append(None)

        print(f"✅ 预加载完成。当前样本总数: {len(self.audio_files)}")



        # 组件初始化
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(xlmr_path)
        self.augmentor = augmentor if augmentor else TicsAugmentation(mode='none')
        self.label_gen = BoundaryLabelGenerator(fps=50)

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        try:
            #print(f"DEBUG: [Rank {torch.distributed.get_rank()}] Loading Index: {idx}")
            audio_path = self.audio_files[idx]
            json_data = self.cached_json[idx]
            # --- 关键调试点：如果还是报错，这里会打印出具体的路径内容 ---
            if json_data is None:
            # 实际生产中建议返回一个 dummy 样本或在 init 中剔除错误
                return self.__getitem__((idx + 1) % len(self))


            # 2. 加载音频
            if not os.path.exists(audio_path):
                return self.__getitem__((idx + 1) % len(self))
                
            waveform, sr = torchaudio.load(audio_path)
             
            # 3. 计算 HuBERT 帧数并过滤超短音频
            target_T = waveform.shape[1] // 320
            if target_T <= 1: 
                return self.__getitem__((idx + 1) % len(self))

            # 4. 生成标签 (传入 json_path 或 meta，取决于你 Generator 的实现)
            y_true = self.label_gen.generate(json_data, target_T)

            # 5. 生成增强视图
            view1 = self.augmentor(waveform, is_view2=False)
            view2 = self.augmentor(waveform, is_view2=True)

            if self.stage == 2:
                text = json_data.get('text', "")
                encoded_text = self.tokenizer(
                    text,
                    padding='max_length',
                    truncation=True,
                    max_length=128,
                    return_tensors='pt'
                )
                return {
                    "view1": view1.squeeze(0),
                    "y_true": y_true,
                    "text_ids": encoded_text['input_ids'].squeeze(0),
                    "text_mask": encoded_text['attention_mask'].squeeze(0)
                }

            return {
                "view1": view1.squeeze(0),
                "view2": view2.squeeze(0),
                "y_true": y_true
            }

        except Exception as e:
            # 万能捕获，确保 400 万训练不会因为某一个 json 格式错误而中断
            # print(f"❌ 索引 {idx} 加载崩溃: {str(e)}")
            return self.__getitem__((idx + 1) % len(self))
        

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