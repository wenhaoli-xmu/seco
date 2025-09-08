import torch
from pygments.console import colorize

baseline = torch.load('test_accuracy/baseline.pth', map_location='cpu')
blockwise = torch.load('test_accuracy/blockwise.pth', map_location='cpu')

print(colorize("red", '=' * 80))
print(colorize("blue", "our chunkwise optimization:"))
print(colorize('yellow', f"{blockwise[0]}"))
print(colorize("blue", "flash attention:"))
print(colorize('green', f"{baseline[0]}"))
print(colorize("red", '=' * 80))
print(colorize("blue", "our chunkwise optimization:"))
print(colorize('yellow', f"{blockwise[-1]}"))
print(colorize("blue", "flash attention:"))
print(colorize('green', f"{baseline[-1]}"))

dist = [torch.dist(x, y) for x, y in zip(baseline, blockwise)]

print('=' * 80)
print('torch.dist()')
for i, x in enumerate(dist):
    if (i + 1) % 10 == 0:
        print(f"{x:.5f}")
    else:
        print(f"{x:.5f}", end='\t')