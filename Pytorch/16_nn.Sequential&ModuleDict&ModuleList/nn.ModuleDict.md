**Sequential, ModuleDict, ModuleList 都继承自 Module**

# nn.ModuleDict 就是Module字典,可以通过key获取Module

> 参数是字典或者二维列表

```python
'''
nn.ModuleDict 就是Module字典,可以通过key获取Module
参数是字典或者二维列表

有update方法,参数可以使字典或者二维列表
def update(self, modules: Mapping[str, Module]) -> None:

'''

import torch
from torch import nn
from collections import OrderedDict


class MyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 添加字典
        self.choices = nn.ModuleDict({
                'conv': nn.Conv2d(3, 16, 3),
        })

        # 有update方法,参数可以使字典或者二维列表
        self.choices.update({'pool': nn.MaxPool2d(3)})

        # 添加二维列表
        self.activations = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()]
        ])

        # 有update方法,参数可以使字典或者二维列表
        self.activations.update([
                ['prelu', nn.PReLU()]
        ])


    def forward(self, x, choice, act):
        x = self.choices[choice](x)
        x = self.activations[act](x)
        return x


x = torch.rand(2, 3, 6, 6)

model = MyModule()

y = model(x, 'conv', 'lrelu')
print(y.size()) # torch.Size([2, 16, 4, 4])

y = model(x, 'pool', 'prelu')
print(y.size()) # torch.Size([2, 3, 2, 2])
```

# ModuleDict继承中的add_module()和items()

> Module的 self.add_module() 和 ModuleDict self.items() 配合起来很好使用

```python
'''
Module的 self.add_module() 和 ModuleDict self.items() 配合起来很好使用
'''


import torch
from torch import nn
from collections import OrderedDict


class MyModule(nn.ModuleDict):
    def __init__(self) -> None:
        super().__init__()
        self.add_module('a', nn.Conv2d(3,  16, 3))
        self.add_module('b', nn.Conv2d(16, 32, 3))


    def forward(self, x):
        for k, v in self.items():
            print(k)
            # a
            # b

            x = v(x)

        return(x)


x = torch.rand(2, 3, 5, 5)

model = MyModule()

y = model(x)
print(y.size()) # torch.Size([2, 32, 1, 1])
```

## densenet示例

```python
class _DenseBlock(nn.ModuleDict):
    '''
    创建DenseBlock,每个DenseBlock都有n个DenseLayer
    将Input放进列表features中,将它作为参数传入DenseLayer,然后将output添加到features列表中,循环放入DenseLayer,最后获得输入+全部输出列表,然后拼接到一起并返回
    最后的输出维度是 out_channel =  input_c + num_layers * growth_rate
    '''
    _version = 2

    def __init__(self,
                 num_layers: int,       # DenseLayer重复次数
                 input_c: int,          # 输出维度
                 bn_size: int,          # 4 growth_rate * bn_size就是DenseLayer中降低的维度
                 growth_rate: int,      # 32 增长率,每一个DenseLayer的输出维度
                 drop_rate: float,
                 memory_efficient: bool = False):
        super().__init__()
        # 构建多个DenseLayer
        for i in range(num_layers):
            layer = _DenseLayer(input_c + i * growth_rate,  # 调整输入的维度,初始维度加上之前所有DenseLayer的输入维度
                                growth_rate=growth_rate,
                                bn_size=bn_size,
                                drop_rate=drop_rate,
                                memory_efficient=memory_efficient)
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        # 将输入放进要放回的列表中
        features = [init_features]

        for name, layer in self.items():
            # 循环使用DenseLayer,参数features是一个数组,所以在DenseLayer中要先拼接数据
            new_features = layer(features)
            # 将输出保存到数组中
            features.append(new_features)

        # 将开始输入和多个层的输出拼接到一起
        return torch.cat(features, 1)
```



