"""
Sequential, ModuleDict, ModuleList 都继承自 Module

Module的 self.add_module() 和 ModuleDict self.items() 配合起来很好使用
self.add_module() 是 Module的方法
self.items() 是ModuleDict的独有方法
"""

import torch
from torch import nn
from collections import OrderedDict


class MyModule(nn.ModuleDict):
    def __init__(self) -> None:
        super().__init__()
        self.add_module("a", nn.Conv2d(3, 16, 3))
        self.add_module("b", nn.Conv2d(16, 32, 3))

    def forward(self, x):
        # self.items() 是ModuleDict的独有方法
        for k, v in self.items():
            print(k)
            # a
            # b

            x = v(x)

        return x


x = torch.rand(2, 3, 5, 5)

model = MyModule()

y = model(x)
print(y.size())  # torch.Size([2, 32, 1, 1])
