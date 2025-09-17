import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# 定义文件名和对应的标签
files_info = {
    '/home/weiweiz/Documents/WW_03/saved/logs/pre_resnet20_MNIST_20_128_0p001_PTQAT_4_1_True_4_1_False_1_False_False.txt': 'QAF',
    '/home/weiweiz/Documents/WW_03/saved/logs/pre+cal+gf2_resnet20_MNIST_20_128_0p001_PTQAT_4_1_True_4_1_False_1_True_True.txt': 'QAF+Cal+GF',
    '/home/weiweiz/Documents/WW_03/saved/logs/pre+cal2_resnet20_MNIST_20_128_0p001_PTQAT_4_1_True_4_1_False_1_False_True.txt': 'QAF+Cal',
    '/home/weiweiz/Documents/WW_03/saved/logs/pre+gf_resnet20_MNIST_20_128_0p001_PTQAT_4_1_True_4_1_False_1_True_False.txt': 'QAF+GF',
    '/home/weiweiz/Documents/WW_03/saved/logs/raw_resnet20_MNIST_20_128_0p001_PTQAT_4_1_True_4_1_False_1_False_False.txt': 'QAT'
}

# 存储所有数据
all_data = {}

# 读取每个文件
for filename, label in files_info.items():
    if os.path.exists(filename):
        data = []
        with open(filename, 'r') as f:
            for line in f:
                if line.strip():  # 跳过空行
                    parts = line.strip().split(', ')
                    epoch = int(parts[0])
                    loss = float(parts[1])
                    train_acc = float(parts[2])
                    test_acc = float(parts[3])
                    data.append([epoch, loss, train_acc, test_acc])
        
        all_data[label] = np.array(data)
        print(f"已读取文件: {filename}, 数据点数: {len(data)}")
    else:
        print(f"文件不存在: {filename}")

# 创建图表
plt.rcParams['font.size'] = 12
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 绘制Loss曲线
ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.grid(True, alpha=0.3)

# 绘制Accuracy曲线
ax2.set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.grid(True, alpha=0.3)

# 定义颜色和线型
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
line_styles = ['-', '--', '-.', ':', '-']

# 为每个实验绘制曲线
for i, (label, data) in enumerate(all_data.items()):
    epochs = data[:, 0]
    losses = data[:, 1]
    test_accuracies = data[:, 3]  # 使用测试准确率
    
    # 绘制loss曲线
    ax1.plot(epochs, losses, 
             color=colors[i % len(colors)], 
             linestyle=line_styles[i % len(line_styles)],
             marker='o', 
             markersize=4,
             linewidth=2,
             label=label)
    
    # 绘制accuracy曲线
    ax2.plot(epochs, test_accuracies, 
             color=colors[i % len(colors)], 
             linestyle=line_styles[i % len(line_styles)],
             marker='s', 
             markersize=4,
             linewidth=2,
             label=label)

# 添加图例
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 设置y轴范围（可选）
ax1.set_ylim(bottom=0)
ax2.set_ylim(0, 100)

# 调整布局
plt.tight_layout()

# 显示图表
plt.show()

# 打印最终结果统计
print("\n=== 最终结果统计 ===")
for label, data in all_data.items():
    final_loss = data[-1, 1]
    final_test_acc = data[-1, 3]
    best_test_acc = np.max(data[:, 3])
    print(f"{label:15} | 最终Loss: {final_loss:.4f} | 最终测试准确率: {final_test_acc:.2f}% | 最佳测试准确率: {best_test_acc:.2f}%")


# plt.savefig('training_results_comparison.png', dpi=300, bbox_inches='tight')
# print("\n图表已保存为 training_results_comparison.png")