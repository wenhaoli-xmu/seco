import matplotlib.pyplot as plt
import numpy as np

# 适当提高饱和度的莫兰蒂色系
morandi_colors = [
    "#737373",  # 深灰色（SeCO）
    "#DD755E",  # 深棕色（SpaCO t=8）
    "#4472B0",  # 深蓝灰色（SpaCO t=16）
    "#6BAC46",  # 深绿灰色（SpaCO t=32）
]

# 数据
learning_rates = np.array([0.0001, 0.0003, 0.001, 0.0016, 0.0023, 0.003, 0.01])
seco = [2.5276, 2.1776, 2.1461, 2.1648,	2.2061, 2.4117, 11.63]
spaco_8 = [2.9197, 2.4760, 2.2916, 2.2981, 2.2816, 2.3135, 5.1]
spaco_16 = [2.6957, 2.3748, 2.2434, 2.2243, 2.2272, 2.2372, 6.9514]
spaco_32 = [2.6126, 2.3085, 2.1924, 2.1807,	2.1929, 2.5568, 7.0178]

# 处理缺失值（去除 None）
def clean_data(x, y):
    x_clean = [x[i] for i in range(len(y)) if y[i] is not None]
    y_clean = [y[i] for i in range(len(y)) if y[i] is not None]
    return x_clean, y_clean

x_seco, y_seco = clean_data(learning_rates, seco)
x_spaco_32, y_spaco_32 = clean_data(learning_rates, spaco_32)

# 画图
plt.figure(figsize=(8, 6))

plt.plot(x_seco, y_seco, marker='^', color=morandi_colors[0], label='Exact Gradient', linestyle='-', linewidth=1.8)
plt.plot(learning_rates, spaco_8, marker='o', color=morandi_colors[1], label='SpaCO t=8', linestyle='-', linewidth=1.8)
plt.plot(learning_rates, spaco_16, marker='o', color=morandi_colors[2], label='SpaCO t=16', linestyle='-', linewidth=1.8)
plt.plot(x_spaco_32, y_spaco_32, marker='o', color=morandi_colors[3], label='SpaCO t=32', linestyle='-', linewidth=1.8)

# X 轴设置为对数刻度
plt.xscale('log')

# 轴标签
plt.xlabel("Learning Rate", fontsize=14)
plt.ylabel("LM Loss", fontsize=14)

# 限制 Y 轴范围
plt.ylim(2, 3)
plt.xlim(1e-4,1e-2)

# 图例
plt.legend(fontsize=12, loc='best')

# 网格
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)


# 显示图表
plt.savefig("lr.jpg", dpi=960)