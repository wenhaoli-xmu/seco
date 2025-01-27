import matplotlib.pyplot as plt
from solos.utils import average_filter
import argparse
import json
import os

# 设置 argparse 以接受多个文件路径
parser = argparse.ArgumentParser()
parser.add_argument("paths", type=str, nargs='+', help="Paths to the data files")
parser.add_argument("--baseline", type=str, required=False)
args = parser.parse_args()

plt.figure()

if args.baseline is not None:
    with open(os.path.join("test_curve", args.baseline), 'r') as f:
        baseline = json.load(f)

# 遍历所有文件路径
for path in args.paths:
    with open(os.path.join("test_curve", path), 'r') as f:
        data = json.load(f)
    # 绘制曲线
    if args.baseline is not None:
        data = [y-x for x, y in zip(baseline, data)]
    plt.plot(average_filter(data, 64), label=path)

# 添加图例
plt.legend()

# 保存图像
plt.savefig(os.path.join("test_curve", "curve.jpg"))