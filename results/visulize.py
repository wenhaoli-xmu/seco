import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
import os

# 设置 argparse 以接受多个文件路径
parser = argparse.ArgumentParser()
parser.add_argument("paths", type=str, nargs='+', help="Paths to the data files")
parser.add_argument("--baseline", type=str, required=False)
args = parser.parse_args()

# 莫兰迪配色（高级配色）
morandi_colors = [
    "#8C8C8C",  # 灰色
    "#D9A688",  # 浅棕色
    "#A3B4C8",  # 浅蓝色
    "#B7C3A3",  # 浅绿色
    "#D4B2A7",  # 粉棕色
    "#C8A8A8",  # 粉红色
]

# 指数滑动平均函数
def exponential_moving_average(data, alpha=0.05):
    ema = np.zeros_like(data)
    ema[0] = np.mean(data[:10])  # 初始值为第一个数据点
    for t in range(1, len(data)):
        ema[t] = alpha * data[t] + (1 - alpha) * ema[t - 1]
    return ema

plt.figure()

# 遍历所有文件路径
for idx, path in enumerate(args.paths):
    with open(os.path.join("results", path), 'r') as f:
        data = json.load(f)
    
    # 将数据转换为 numpy 数组
    data = np.array(data)  # 形状为 [N, L]
    
    # 对每条曲线进行指数滑动平均
    data_ema = np.array([exponential_moving_average(curve, alpha=0.01) for curve in data])
    
    # 计算均值和标准差
    mean_data = np.mean(data_ema, axis=0)  # 形状为 [L,]
    std_data = np.std(data_ema, axis=0)    # 形状为 [L,]
    
    # 绘制均值曲线
    plt.plot(mean_data, color=morandi_colors[idx], label=f'{path} (Mean)')
    
    # 绘制标准差区域
    plt.fill_between(
        range(len(mean_data)),  # x 轴
        mean_data - std_data,   # y1
        mean_data + std_data,   # y2
        color=morandi_colors[idx],  # 颜色
        alpha=0.2,              # 透明度
        label=f'{path} (±1 Std Dev)'  # 图例
    )

# 添加图例
plt.legend()
plt.xlim(0, 999)

# 保存图像
plt.savefig(os.path.join("results", "curve_with_std_ema.jpg"))