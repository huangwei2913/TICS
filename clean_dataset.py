import os
import torch
import torchaudio
import glob
from tqdm import tqdm
import shutil
from multiprocessing import Pool, cpu_count
import functools

def validate_pt_file(file_path, src_dir, bad_dir, max_k=6, min_duration=0.5, max_duration=30.0):
    """
    验证单个 .pt 文件，如果损坏则直接移动
    """
    try:
        # 1. 基础读取
        data = torch.load(file_path, map_location='cpu')
        
        # 2. 文本校验
        text = data.get('text', "")
        if not text or len(str(text).strip()) == 0:
            return move_file(file_path, src_dir, bad_dir, "Empty_Text")

        # 3. 音频路径校验
        wav_path = data.get('wav_path', "")
        if not wav_path or not os.path.exists(wav_path):
            return move_file(file_path, src_dir, bad_dir, "Missing_Wav")

        # 4. K 值校验 (句子计数)
        target_k = data.get('segment_count_nltk', 0)
        if target_k <= 0 or target_k > max_k:
            return move_file(file_path, src_dir, bad_dir, f"Invalid_K_{target_k}")

        # 5. 音频长度校验 (关键：info 比 load 快得多)
        info = torchaudio.info(wav_path)
        duration = info.num_frames / info.sample_rate
        if duration < min_duration or duration > max_duration:
            return move_file(file_path, src_dir, bad_dir, f"Invalid_Duration_{duration:.1f}s")

        # 6. 特征完整性
        if 'boundary' not in data or 'emotion_feats' not in data:
            return move_file(file_path, src_dir, bad_dir, "Missing_Features")

        return "OK"

    except Exception as e:
        return move_file(file_path, src_dir, bad_dir, f"Crash_{type(e).__name__}")

def move_file(file_path, src_dir, bad_dir, reason):
    """移动坏文件并保持目录结构"""
    try:
        rel_path = os.path.relpath(file_path, src_dir)
        dest_path = os.path.join(bad_dir, rel_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.move(file_path, dest_path)
        return reason
    except Exception:
        return "Move_Failed"

def main():
    # --- 配置区 ---
    SRC_DIR = "/data/NaturalVoices/Stage2_Features"
    BAD_DIR = "/data/NaturalVoices/Stage2_Features_Bad"
    NUM_WORKERS = cpu_count() # 使用全部 CPU 核心
    # --------------

    print(f"Scanning files in {SRC_DIR}...")
    pt_files = glob.glob(os.path.join(SRC_DIR, "**/*.pt"), recursive=True)
    total_files = len(pt_files)
    print(f"Found {total_files} files. Starting parallel cleaning with {NUM_WORKERS} workers...")

    # 使用偏函数固定参数
    worker_func = functools.partial(
        validate_pt_file, 
        src_dir=SRC_DIR, 
        bad_dir=BAD_DIR, 
        max_k=6
    )

    # 结果统计
    results_stats = {}

    with Pool(NUM_WORKERS) as pool:
        # imap 会返回一个迭代器，配合 tqdm 显示进度
        for res in tqdm(pool.imap_unordered(worker_func, pt_files), total=total_files):
            results_stats[res] = results_stats.get(res, 0) + 1

    # 打印总结报告
    print("\n" + "="*30)
    print("      CLEANING REPORT")
    print("="*30)
    ok_count = results_stats.pop("OK", 0)
    print(f"✅ VALID FILES: {ok_count}")
    print(f"❌ BAD FILES:   {total_files - ok_count}")
    print("-" * 30)
    print("REASONS FOR REMOVAL:")
    for reason, count in sorted(results_stats.items(), key=lambda x: x[1], reverse=True):
        print(f" - {reason}: {count}")
    print("="*30)

if __name__ == "__main__":
    main()