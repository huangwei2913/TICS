import pandas as pd
import os
import json
import torchaudio
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import logging

# 1. 配置：只记录真正坏掉的样本
logging.basicConfig(filename='data_error_report.log', level=logging.ERROR, 
                    format='%(message)s')

def check_row(row_tuple):
    """
    输入 row_tuple: (index, wav_path, json_path)
    返回: 如果正常返回 (index, True), 如果损坏返回 (index, False)
    """
    idx, wav, json_p = row_tuple
    
    # 基本检查：是否是路径格式
    if not isinstance(wav, str) or not wav.startswith('/'):
        return idx, False

    try:
        # 核心校验 A: 音频头部信息读取 (不加载数据，极快)
        # 如果文件损坏，torchaudio.info 会抛出异常
        info = torchaudio.info(wav)
        if info.num_frames == 0:
            logging.error(f"Empty Wav|{idx}|{wav}")
            return idx, False
        
        # 核心校验 B: JSON 格式校验
        with open(json_p, 'r') as f:
            json.load(f)
            
    except Exception as e:
        logging.error(f"Corrupt|{idx}|{wav}|{str(e)}")
        return idx, False
    
    return idx, True

def main():
    input_csv = "/home/huangwei/TICS/valid_samples_wav_json.csv"
    output_csv = "/home/huangwei/TICS/cleaned_samples_wav_json.csv"
    
    print(f"--- 启动 1.2M 数据全量清洗 ---")
    
    # 1. 快速加载 CSV (只取前两列)
    df = pd.read_csv(input_csv, header=None, usecols=[0, 1])
    total = len(df)
    print(f"📊 原始数据总量: {total}")

    # 2. 准备并行任务
    # 使用 zip 组合数据，避免在大循环中使用 df.iloc，能显著提升速度
    tasks = list(zip(df.index, df[0], df[1]))

    # 3. 开启多进程执行 (推荐 workers 设为 CPU 核心数的 80%)
    valid_indices = []
    print(f"🚀 正在并行校验 (详情查看 data_error_report.log)...")
    
    # chunksize=500 能平衡进程间切换的开销
    with ProcessPoolExecutor(max_workers=12) as executor:
        results = list(tqdm(executor.map(check_row, tasks, chunksize=500), total=total))

    # 4. 筛选正常样本
    valid_indices = [idx for idx, is_valid in results if is_valid]
    
    # 5. 生成新 CSV
    print(f"💾 正在保存干净的 CSV...")
    df_cleaned = df.loc[valid_indices]
    df_cleaned.to_csv(output_csv, index=False, header=False)
    
    print("\n" + "="*40)
    print(f"✅ 清洗完成！")
    print(f"📦 原始样本: {total}")
    print(f"✨ 干净样本: {len(df_cleaned)}")
    print(f"🗑️ 剔除坏账: {total - len(df_cleaned)}")
    print(f"📄 新文件已保存至: {output_csv}")
    print("="*40)

if __name__ == "__main__":
    main()