import json
import argparse
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument('--baseline', type=str, default='blockwise-tp.pth')
parser.add_argument('--ours', type=str, default='blockwise-tp-sparse.pth')
parser.add_argument('--root-dir', type=str, default='test_accuracy')
args = parser.parse_args()


baseline = torch.load(os.path.join(args.root_dir, args.baseline), map_location='cpu')
ours = torch.load(os.path.join(args.root_dir, args.ours), map_location='cpu')

name = baseline['name']
baseline = baseline['grad']
ours = ours['grad']

for n, b, o in zip(name, baseline, ours):
    print(f"{n:<144}{torch.dist(b, o)}")