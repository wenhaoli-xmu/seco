import torch

reference = torch.load("grads/oracle.pth", map_location='cpu')
seco = torch.load("grads/seco.pth", map_location='cpu')

diff = torch.dist(reference, seco)
print(f"difference: {diff}")
