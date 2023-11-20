#-------------------------------------------------------------------------------------#
#   self.register_buffer('name', Tensor)的操作，该方法的作用是定义一组参数，
#   该组参数的特别之处在于：模型训练时不会更新(即调用 optimizer.step() 后该组参数不会变化，
#   只可人为地改变它们的值，但是保存模型时，该组参数又作为模型参数不可或缺的一部分被保存。
#-------------------------------------------------------------------------------------#


import torch
from torch import nn as nn


class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        # (1) 常见定义模型时的操作
        self.fc = nn.Linear(10, 10)

        # (2) 使用Parameter定义一组参数
        self.se = nn.Parameter(torch.randn(1, 2))

        # (3) 使用形式类似的 register_parameter() 定义一组参数，和上面效果相同
        self.register_parameter('param_reg', nn.Parameter(torch.randn(1, 2)))

        # (4) 使用 register_buffer() 定义一组参数,不会被梯度更新
        self.register_buffer('param_buf', torch.randn(1, 2))

        # (5) 按照类的属性形式定义一组变量，不会被state_dict()保存
        self.param_attr = torch.randn(1, 2)

    def forward(self, x):
        pass


model = Model()
print(model.state_dict().keys())
# odict_keys(['se', 'param_reg', 'param_buf', 'fc.weight', 'fc.bias'])
# 看到有定义的前四个变量的参数，没有直接使用self保存的参数