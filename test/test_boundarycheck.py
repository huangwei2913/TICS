import torch
import json
import matplotlib.pyplot as plt

class BoundaryLabelGenerator:
    def __init__(self, fps: int = 50):
        self.fps = fps

    def json_to_ytrue(self, json_path: str, target_frames: int) -> torch.Tensor:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        y_true = torch.zeros(target_frames, dtype=torch.float32)
        words = data.get('words', data.get('word_segments', []))

        for word in words:
            end_time = word.get('end', None)
            if end_time is None:
                continue
            frame_idx = int(round(end_time * self.fps))
            if 0 <= frame_idx < target_frames:
                y_true[frame_idx] = 1.0

        return y_true

# ----------------------------
# 1. 配置参数
# ----------------------------
json_path = "/home/huangwei/TICS/test/ZH_B00089_S09915_W000001.json"        # 你的 JSON 文件路径
target_frames = 100             # 假设模型输出有 100 帧

# ----------------------------
# 2. 生成 y_true
# ----------------------------
label_gen = BoundaryLabelGenerator(fps=50)
y_true = label_gen.json_to_ytrue(json_path, target_frames)

# ----------------------------
# 3. 打印基本信息
# ----------------------------
print("y_true shape:", y_true.shape)
print("y_true dtype:", y_true.dtype)
print("Number of boundary points:", y_true.sum().item())

# 打印所有为 1 的位置
boundary_indices = torch.nonzero(y_true).squeeze().tolist()
if isinstance(boundary_indices, int):
    boundary_indices = [boundary_indices]
print("Boundary frame indices:", boundary_indices)

# 打印每个边界对应的时间（秒）
boundary_times = [idx / 50.0 for idx in boundary_indices]
print("Boundary times (seconds):", [f"{t:.3f}" for t in boundary_times])

# ----------------------------
# 4. 可视化 y_true
# ----------------------------
plt.figure(figsize=(12, 3))
plt.plot(y_true.numpy(), 'o-', markersize=4, alpha=0.8)
plt.title("Generated y_true (Boundary Labels)")
plt.xlabel("Frame Index")
plt.ylabel("Label (0/1)")
plt.grid(True, alpha=0.3)
plt.ylim(-0.1, 1.1)
plt.show()
