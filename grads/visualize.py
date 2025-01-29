import matplotlib.pyplot as plt
import os
import json

import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("paths", type=str, nargs='+', help="Paths to the data files")
parser.add_argument("--baseline", type=str, required=True)
args = parser.parse_args()


# 初始化绘图
plt.figure(figsize=(10, 7))

# 加载基准数据
baseline = None
if args.baseline is not None:
    baseline = torch.load(f"grads/{args.baseline}", map_location='cpu')
    baseline = baseline.mean(0, keepdim=True)


indices = torch.randperm(baseline.numel())[:100]


# 遍历所有文件路径
for i, path in enumerate(args.paths):
    data = torch.load(f"grads/{path}", map_location='cpu')
    if baseline is not None:
        data -= baseline

    data = data.mean(0)[indices]
    plt.plot(list(range(data.numel())), data.tolist(), label=f"{path}")
    

# 添加网格、标题和图例
# # plt.ylim(-0.0005, 0.0005)
# plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
# plt.xticks([])
# # plt.ylim(0, 8)
# # plt.legend(loc='lower right')
# plt.grid(True, linestyle='--', alpha=0.6)


# 保存图像
plt.legend()
plt.savefig(os.path.join("grads", "visualize.jpg"), dpi=960)

