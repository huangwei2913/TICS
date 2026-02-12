import torch
import librosa
import numpy as np
from moco_tics.TICSMoCo import TICS_MoCo

# --- 1. 定义与训练一致的配置 ---
TEACHER_CONFIG = {'input_dim': 768, 'segment_dim': 1024, 'num_layers': 12}

# --- 2. 加载模型 ---
# 传入 teacher_config 解决 TypeError
model = TICS_MoCo(
    backbone_path="/mnt/facebook/hubert-base-ls960", 
    teacher_config=TEACHER_CONFIG
)

# 注意：如果你的权重保存的是 state_dict，直接 load
# 如果报错，可能是因为转换出的 .pt 放在了文件夹里，路径请指向到具体的 .bin 或 .pt 文件
checkpoint_path = "/data/tics_training/backups/final_stage1_step13w.pt/pytorch_model.bin"
state_dict = torch.load(checkpoint_path, map_location="cpu")

model.load_state_dict(state_dict)
model.eval()
print("✅ 模型加载成功！")

# --- 3. 读取测试音频 (16k 采样率) ---
audio_path = "dia0_utt7.wav"
wav, _ = librosa.load(audio_path, sr=16000)
wav_tensor = torch.from_numpy(wav).unsqueeze(0) # [1, T]

# --- 4. 前向计算 ---
with torch.no_grad():
    # 执行模型，推理时 q 和 k 输入同一个 wav 即可
    outputs = model(wav_tensor, wav_tensor, aug_mode="none")
    # 提取 P_score
    p_scores = outputs["P_score"].squeeze().cpu().numpy()

# --- 5. 结果观察 ---
# 打印前20个数值看看
print("P_scores 前20个采样点预览:")
print(p_scores[:20])

# 如果你想看有没有波峰（边界）：
max_val = np.max(p_scores)
mean_val = np.mean(p_scores)
print(f"最大值: {max_val:.4f}, 平均值: {mean_val:.4f}")

if max_val > 0.5:
    print("🚀 发现明显边界信号！模型确实学会了切分。")
else:
    print("归一化提示：如果数值都很小，可能是因为还在 Sigmoid 前或者还没遇到停顿。")