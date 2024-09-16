"""
Sequential, ModuleDict, ModuleList 都继承自 Module
"""

import torch
from torch import nn
from collections import OrderedDict


x = torch.rand(2, 1, 224, 224)

# Sequential传入单个数据
model = nn.Sequential(nn.Conv2d(1, 20, 5), nn.ReLU(), nn.Conv2d(20, 64, 5), nn.ReLU())

# Sequential可以在后面添加模型
model.add_module(name="n", module=nn.Conv2d(64, 128, 5))
y = model(x)
print(y.size())  # [2, 128, 212, 212]

# Sequential传入OrderedDict
model = nn.Sequential(
    OrderedDict(
        [
            ("conv1", nn.Conv2d(1, 20, 5)),
            ("relu1", nn.ReLU()),
            ("conv2", nn.Conv2d(20, 64, 5)),
            ("relu2", nn.ReLU()),
        ]
    )
)

y = model(x)
print(y.size())  # [2, 64, 216, 216]
