import json
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("path", type=str)
args = parser.parse_args()


with open(f"results/{args.path}", 'r') as f:
    data = json.load(f)


for x in data:
    mean_value = np.mean(x[-100:])
    print(mean_value)
