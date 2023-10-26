'''
Sequential, ModuleDict, ModuleList 都继承自 Module

继承其实就是将Module列表拆开或者OrderedDict直接放进nn.Sequential的__init__中
'''

import torch
from torch import nn
from collections import OrderedDict


class MySequential1(nn.Sequential):

    def __init__(self) -> None:

        layer = []
        layer.append(nn.Conv2d(3, 16, 3, padding=1))
        layer.append(nn.Conv2d(16, 32, 3, padding=1))

        # 将列表拆开放进__init__
        super().__init__(*layer)



class MySequential2(nn.Sequential):

    def __init__(self) -> None:

        layer = OrderedDict()

        # 有序字典的两种添加方式
        layer['a'] = nn.Conv2d(32, 64, 3, padding=1)
        layer.update({'b': nn.Conv2d(64, 128, 3, padding=1)})

        # 将有序字典直接放进__init__
        super().__init__(layer)


class MyModule(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        # 和正常的列表使用方式相同
        self.layer1 = MySequential1()
        self.layer2 = MySequential2()


    def forward(self, x):

        x = self.layer1(x)
        print(x.size())     # torch.Size([2, 32, 224, 224])
        x = self.layer2(x)
        print(x.size())     # torch.Size([2, 128, 224, 224])


x = torch.rand(2, 3, 224, 224)
model = MyModule()
y = model(x)