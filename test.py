import torch
import torch.nn.functional as F

# Create a tensor with logits (raw scores)
logits_2d = torch.tensor([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0]])

logits_3d = torch.tensor([[[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0]],
                          [[0.1, 0.2, 0.3],
                           [0.4, 0.5, 0.6]]])

# Apply softmax along the last dimension using dim=-1
softmax_2d = F.softmax(logits_2d, dim=-1)
softmax_3d = F.softmax(logits_3d, dim=-1)

print("Logits (2D):")
print(logits_2d)

print("Softmax Output (2D):")
print(softmax_2d)

print("Logits (3D):")
print(logits_3d)

print("Softmax Output (3D):")
print(softmax_3d)
