import torch
from torch import nn
from torch.nn import functional as F

t3 = torch.ones(4, 3, 2, 3)
t4 = torch.rand(4, 3, 3, 5)

print((t3@t4).size())
# torch.Size([4, 3, 2, 5])  # 前两个维度不变,后两个维度相乘


x = torch.tensor([
    [1, 4],
    [5, 5.]
])

print(torch.softmax(x, dim=-1))
# [[0.0474, 0.9526],
#  [0.5000, 0.5000]]


x = torch.randn(1, 1, 224, 224)
conv = nn.Conv2d(1, 1, 7, stride=4, padding=3)  # [1, 1, 56, 56]
print(conv(x).size())
conv = nn.Conv2d(1, 1, 7, stride=4, padding=2)  # [1, 1, 56, 56]
print(conv(x).size())
