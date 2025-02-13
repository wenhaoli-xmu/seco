import torch

def hook(grad):
    return grad + 1


a = torch.tensor([1], dtype=torch.float, requires_grad=True)
b = (a + 2) ** 2
b.register_hook(hook)

b.backward()

import IPython
IPython.embed()