import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_training_results(csv_path):
    # 1. 加载数据
    df = pd.read_csv(csv_path)
    
    # 设置绘图风格
    sns.set_theme(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # 2. 绘制 Loss 曲线 (左轴)
    # 提取不同的 Loss
    total_loss = df[df['Metric'] == 'Loss/Total']
    moco_loss = df[df['Metric'] == 'Loss/MoCo']
    sup_loss = df[df['Metric'] == 'Loss/Supervised']

    # 绘制并平滑处理 (使用移动平均让曲线更好看)
    window = 50 # 平滑窗口大小
    ax1.plot(total_loss['Step'], total_loss['Value'].rolling(window=window).mean(), 
             label='Total Loss', color='#E74C3C', linewidth=2)
    ax1.plot(moco_loss['Step'], moco_loss['Value'].rolling(window=window).mean(), 
             label='MoCo Loss', color='#3498DB', linestyle='--', alpha=0.7)
    ax1.plot(sup_loss['Step'], sup_loss['Value'].rolling(window=window).mean(), 
             label='Supervised Loss', color='#2ECC71', linestyle='--', alpha=0.7)

    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Loss Value', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.set_title('Stage 1 Training Dynamics (1.23M Data / 82 Hours)', fontsize=15)

    # 3. 绘制 P_avg 曲线 (右轴 - 共享 X 轴)
    ax2 = ax1.twinx()
    p_avg = df[df['Metric'] == 'Stats/P_avg']
    ax2.plot(p_avg['Step'], p_avg['Value'].rolling(window=window).mean(), 
             label='Boundary Probability (P_avg)', color='#F1C40F', linewidth=2)
    
    ax2.set_ylabel('Avg Prediction Probability', fontsize=12, color='#F39C12')
    ax2.tick_params(axis='y', labelcolor='#F39C12')
    ax2.set_ylim(0, 0.5) # P_avg 正常范围在 0-0.5 之间
    ax2.legend(loc='upper right')

    # 保存图片
    plt.tight_layout()
    plt.savefig('stage1_analysis.png', dpi=300)
    plt.show()

# 运行绘图
plot_training_results("stage1_training_curve.csv")