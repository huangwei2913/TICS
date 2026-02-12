import pandas as pd

# 原始大文件路径
large_csv = "/home/huangwei/emilia_en_ch_training_final.csv"
# 微型测试文件路径
mini_csv = "/home/huangwei/emilia_mini_test.csv"

# 读取前 1000 条（或者随机抽取）
df = pd.read_csv(large_csv)
mini_df = df.sample(n=1000, random_state=42) # 随机抽取1000条

# 保存
mini_df.to_csv(mini_csv, index=False)
print(f"✅ 已生成微型测试集：{mini_csv}，共 {len(mini_df)} 条记录。")