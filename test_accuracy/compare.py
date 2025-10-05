import torch
from pygments.console import colorize
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--baseline', type=str, default='baseline-tp.pth')
parser.add_argument('--ours', type=str, default='baseline-tp-sparse.pth')
parser.add_argument('--root-dir', type=str, default='test_accuracy')
args = parser.parse_args()

baseline = torch.load(os.path.join(args.root_dir, args.baseline), map_location='cpu')
if isinstance(baseline, dict):
    baseline = baseline['grad']

files = os.listdir(args.root_dir)
files.remove(args.baseline)
files = list(filter(lambda x: x.endswith('.pth'), files))

dists = []

print(colorize("red", '=' * 80))
print(colorize("cyan", 'Distance from baseline: torch.dist(baseline, ours)'))

for file in files:
    ours = torch.load(os.path.join(args.root_dir, file), map_location='cpu')
    if isinstance(ours, dict):
        ours = ours['grad']

    distance = [torch.dist(x, y) for x, y in zip(baseline[:-1], ours[:-1])]
    dists.append(distance)

    title = f"{file}"[-18:]
    postfix = ' ' * (20 - len(title))
    print(colorize("green", title + postfix), end='', flush=True)
print()

for i in range(len(dists[0])):
    for j in range(len(files)):
        print(f"{dists[j][i]:<20.5f}", end='')
    print()

print(colorize("red", '=' * 80))
