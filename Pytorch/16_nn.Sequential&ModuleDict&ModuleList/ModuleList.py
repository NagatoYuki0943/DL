"""
Sequential, ModuleDict, ModuleList 都继承自 Module

nn.ModuleList 就是Module列表,可以迭代获取
可以通过下标或者直接调用
ModuleList无法执行,只能取出来单独执行

像普通列表一样有下列方法

def insert(self, index: int, module: Module) -> None:
    pass

def append(self, module: Module) -> 'ModuleList':
    pass

def extend(self, modules: Iterable[Module]) -> 'ModuleList':
    pass
"""

import torch
from torch import nn
from collections import OrderedDict


class MyModule1(nn.Module):
    def __init__(self):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.linears):
            print(i, l)
            # 0 Linear(in_features=10, out_features=10, bias=True)
            # 1 Linear(in_features=10, out_features=10, bias=True)

            # 可以通过下标或者直接调用
            x1 = self.linears[i // 2](x)
            x2 = l(x)
            x = x1 + x2

        return x


x = torch.rand(2, 10)
model = MyModule1()
y = model(x)
print(y.size())  # torch.Size([2, 10])


# --------------------------------------------#
#   Sequential可以执行
# --------------------------------------------#
x = torch.rand(2, 3, 128, 128)
model = nn.Sequential(*[nn.Conv2d(3, 64, 1), nn.BatchNorm2d(64), nn.ReLU()])
y = model(x)
print(y.size())  # torch.Size([2, 64, 128, 128])


# --------------------------------------------#
#   ModuleList无法执行,只能取出来单独执行
# --------------------------------------------#
x = torch.rand(2, 3, 128, 128)
model = nn.ModuleList([nn.Conv2d(3, 64, 1), nn.BatchNorm2d(64), nn.ReLU()])
y = model(x)
print(y.size())  # 报错
