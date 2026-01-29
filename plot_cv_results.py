import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 超图神经网络的10折交叉验证结果
fold_results = {
    'Fold': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'ACC': [72.64, 68.87, 65.71, 65.71, 67.62, 62.86, 60.95, 62.86, 65.71, 61.90],
    'SEN': [72.92, 56.41, 57.45, 62.07, 68.09, 60.42, 62.71, 64.71, 75.47, 52.83],
    'SPE': [72.41, 76.12, 72.41, 70.21, 67.24, 64.91, 58.70, 61.11, 55.77, 71.15],
    'PPV': [68.63, 57.89, 62.79, 72.00, 62.75, 59.18, 66.07, 61.11, 63.49, 65.12],
    'NPV': [76.36, 75.00, 67.74, 60.00, 72.22, 66.07, 55.10, 64.71, 69.05, 59.68],
    'F1': [72.78, 62.02, 61.30, 63.84, 67.85, 61.61, 61.82, 63.77, 70.26, 57.01],
    'AUC': [74.60, 70.95, 68.20, 68.56, 72.19, 68.53, 66.32, 70.59, 77.61, 74.31]
}

df = pd.DataFrame(fold_results)

# 设置中文字体（避免中文显示问题）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

metrics = ['ACC', 'SEN', 'SPE', 'PPV', 'NPV', 'F1', 'AUC']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

# 为每个指标单独生成一张图
for idx, metric in enumerate(metrics):
    # 创建新图形
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制折线图（转换为numpy数组）
    ax.plot(df['Fold'].values, df[metric].values, marker='o', linewidth=2.5, markersize=10, 
            color=colors[idx], label=metric, markerfacecolor=colors[idx], 
            markeredgecolor='white', markeredgewidth=2)
    
    # 添加平均线
    mean_value = df[metric].mean()
    ax.axhline(y=mean_value, color=colors[idx], linestyle='--', linewidth=2, 
               alpha=0.7, label=f'Mean: {mean_value:.2f}%')
    
    # 添加标准差阴影
    std_value = df[metric].std()
    ax.fill_between(df['Fold'].values, mean_value - std_value, mean_value + std_value, 
                     color=colors[idx], alpha=0.2, label=f'±1 Std: {std_value:.2f}%')
    
    # 在每个点上标注数值
    for i, (fold, value) in enumerate(zip(df['Fold'].values, df[metric].values)):
        ax.annotate(f'{value:.1f}', 
                   xy=(fold, value), 
                   xytext=(0, 10), 
                   textcoords='offset points',
                   ha='center', 
                   fontsize=9,
                   color=colors[idx],
                   weight='bold')
    
    # 设置标题和标签
    ax.set_title(f'{metric} Performance across 10 Folds - Hypergraph Neural Network', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Fold Number', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{metric} (%)', fontsize=12, fontweight='bold')
    ax.set_xticks(df['Fold'].values)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    
    # 设置y轴范围
    ax.set_ylim([0, 100])
    
    # 美化边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    filename = f'D:/Braingnn/超图/cv_results_{metric}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"已保存: {filename}")
    
    # 关闭图形以释放内存
    plt.close()

# 打印统计信息
print("\n" + "="*60)
print("超图神经网络 - 10-Fold Cross Validation Statistics:")
print("="*60)
for metric in metrics:
    mean_val = df[metric].mean()
    std_val = df[metric].std()
    min_val = df[metric].min()
    max_val = df[metric].max()
    print(f"{metric:4s}: {mean_val:5.2f}±{std_val:4.2f}% (Min: {min_val:5.2f}%, Max: {max_val:5.2f}%)")
print("="*60)
print(f"\n所有图片已保存到: D:/Braingnn/超图/")
