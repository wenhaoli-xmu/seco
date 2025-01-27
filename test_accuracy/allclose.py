import os
import argparse
import torch

from pygments.console import colorize



parser = argparse.ArgumentParser()
parser.add_argument("grad1", type=str)
parser.add_argument("grad2", type=str)
parser.add_argument("--debug", action='store_true')
args = parser.parse_args()


path1 = os.path.join("test_accuracy", "grads", args.grad1)
grad1 = torch.load(path1)

path2 = os.path.join("test_accuracy", "grads", args.grad2)
grad2 = torch.load(path2)



for gd1, gd2 in zip(grad1, grad2):
    assert gd1.shape == gd2.shape

    shape = f"{gd1.shape}"
    
    allclose = torch.dist(gd1, gd2) < 0.1

    if allclose:
        print(f"{shape:<30}" + colorize("green", "pass"))
    else:
        print(f"{shape:<30}" + colorize("red", "not pass"))

    if args.debug:
        import IPython  
        IPython.embed()