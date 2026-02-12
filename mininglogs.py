import os
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd

def extract_tensorboard_data(log_dir, output_csv):
    # 自动定位文件夹里那个 733K 的文件
    event_file = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if 'events.out' in f][0]
    
    # 加载数据
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    
    # 获取所有的 tag (比如 'Loss/Total', 'Stats/P_avg')
    tags = ea.Tags()['scalars']
    
    all_data = []
    for tag in tags:
        # 提取指定 tag 的时间戳、步数和数值
        for event in ea.Scalars(tag):
            all_data.append({
                'Metric': tag,
                'Step': event.step,
                'Value': event.value,
                'Wall_time': event.wall_time
            })
    
    # 转为 DataFrame 并保存
    df = pd.DataFrame(all_data)
    df.to_csv(output_csv, index=False)
    print(f"✅ 曲线数据已成功提取至: {output_csv}")

# 使用示例
extract_tensorboard_data("/data/tics_training/logs", "stage1_training_curve.csv")