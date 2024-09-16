"""
Sequential, ModuleDict, ModuleList 都继承自 Module

nn.ModuleDict 就是Module字典,可以通过key获取Module
参数是字典或者二维列表

有update方法,参数可以使字典或者二维列表
def update(self, modules: Mapping[str, Module]) -> None:

"""

import torch
from torch import nn
from collections import OrderedDict


class MyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 添加字典
        self.choices = nn.ModuleDict(
            {
                "conv": nn.Conv2d(3, 16, 3),
            }
        )

        # 有update方法,参数可以使字典或者二维列表
        self.choices.update({"pool": nn.MaxPool2d(3)})

        # 添加二维列表
        self.activations = nn.ModuleDict([["lrelu", nn.LeakyReLU()]])

        # 有update方法,参数可以使字典或者二维列表
        self.activations.update([["prelu", nn.PReLU()]])

    def forward(self, x, choice, act):
        x = self.choices[choice](x)
        x = self.activations[act](x)
        return x


x = torch.rand(2, 3, 6, 6)

model = MyModule()

y = model(x, "conv", "lrelu")
print(y.size())  # torch.Size([2, 16, 4, 4])

y = model(x, "pool", "prelu")
print(y.size())  # torch.Size([2, 3, 2, 2])
