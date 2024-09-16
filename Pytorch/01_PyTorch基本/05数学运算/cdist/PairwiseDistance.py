import torch
from torch import nn

# 计算两组相同形状数据的距离,默认按照最后通道的全部数据进行计算
# cdist保持column相同即可, PairwiseDistance必须保持形状完全相同
model = nn.PairwiseDistance(p=2)

x = torch.randn(2, 3)
y = torch.randn(2, 3)
z = model(x, y)
print(z.size())  # [2]


x = torch.randn(2, 3, 4)
y = torch.randn(2, 3, 4)
z = model(x, y)
print(z.size())  # [2, 3]     默认在最后维度计算


try:
    # 不同形状报错,与cdist不同
    x = torch.randn(2, 3)
    y = torch.randn(2, 4)
    print(
        "cdist:", torch.cdist(x, y).size()
    )  # X1 and X2 must have the same number of columns. X1: 3 X2: 4
    z = model(x, y)
except Exception as e:
    print(
        e
    )  # The size of tensor a (3) must match the size of tensor b (4) at non-singleton dimension 1

try:
    # 不同形状报错,与cdist不同
    x = torch.randn(2, 3)
    y = torch.randn(3, 3)
    print("cdist:", torch.cdist(x, y).size())  # [2, 3】
    z = model(x, y)
except Exception as e:
    print(
        e
    )  # The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 0
