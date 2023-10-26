import torch

# 演示不同通道的均值
# b*c
a = torch.tensor([
    [1., 3, 5],
    [2,  4, 6],
    [3,  5, 7]
])

# 在每个c上取均值(每一列) BN
print(a.mean(dim=0))    # tensor([2., 4., 6.])
# 在每个b上取均值(每一行) LN
print(a.mean(dim=1))    # tensor([3., 4., 5.])