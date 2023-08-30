import math
import torch

pe = torch.zeros(5,6)

print(pe)

for pos in range(5):
    for i in range(0, 6, 2):
        theta = pos / (10000 ** (i / 6))
        pe[pos, i    ] = math.sin(theta)
        pe[pos, i + 1] = math.cos(theta)

print(pe)