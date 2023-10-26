**Sequential, ModuleDict, ModuleList 都继承自 Module**

# nn.Sequential参数

```python
# Using Sequential to create a small model. When `model` is run,
# input will first be passed to `Conv2d(1,20,5)`. The output of
# `Conv2d(1,20,5)` will be used as the input to the first
# `ReLU`; the output of the first `ReLU` will become the input
# for `Conv2d(20,64,5)`. Finally, the output of
# `Conv2d(20,64,5)` will be used as input to the second `ReLU`
model = nn.Sequential(
            nn.Conv2d(1,20,5),
            nn.ReLU(),
            nn.Conv2d(20,64,5),
            nn.ReLU()
        )

# Using Sequential with OrderedDict. This is functionally the
# same as the above code
model = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1,20,5)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(20,64,5)),
            ('relu2', nn.ReLU())
        ]))
```

# 基本使用

```python
import torch
from torch import nn
from collections import OrderedDict


x = torch.rand(2, 1, 224, 224)

# Sequential传入单个数据
model = nn.Sequential(
            nn.Conv2d(1, 20, 5),
            nn.ReLU(),
            nn.Conv2d(20, 64, 5),
            nn.ReLU()
        )

y = model(x)
print(y.size()) # torch.Size([2, 64, 216, 216])

# Sequential传入OrderedDict
model = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 20, 5)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(20, 64, 5)),
            ('relu2', nn.ReLU())
        ]))

y = model(x)
print(y.size()) # torch.Size([2, 64, 216, 216])
```

# 继承

> 继承其实就是将Module列表拆开或者OrderedDict直接放进nn.Sequential的__init__中

```python
'''
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
```



# Sequential切片

> Sequential,ModuleDict,ModuleList都能使用切片进行分步骤计算

```python
# 13,13,1024 -> 13,13,512
out0_branch = self.last_layer0[:5](x0)              # 前5次特征提取,要上采样拼接
# 13,13,512 -> 13,13,512
out0        = self.last_layer0[5:](out0_branch)     # 后2次利用YoloHead获得预测结果
```





```python
from collections import OrderedDict

import torch
import torch.nn as nn

from nets.darknet import darknet53

# conv+bn+relu
def conv2d(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

#------------------------------------------------------------------------#
#   make_last_layers里面一共有七个卷积，前五个用于提取特征。
#   后两个用于获得yolo网络的预测结果
#------------------------------------------------------------------------#
def make_last_layers(filters_list, in_filters, out_filter):
    """
    filters_list: 维度变化列表
    in_filters:   in_channels
    out_filter:   最终conv的out_channels

    """
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),         # 1 调整通道
        conv2d(filters_list[0], filters_list[1], 3),    # 3 提取特征
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        nn.Conv2d(filters_list[1], out_filter, kernel_size=1, stride=1, padding=0, bias=True)
    )
    return m

class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes):
        """
        anchors_mask: [[6, 7, 8], [3, 4, 5], [0, 1, 2]] 用于帮助代码找到对应的先验框
        """
        super(YoloBody, self).__init__()
        #---------------------------------------------------#
        #   生成darknet53的主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256
        #   26,26,512
        #   13,13,1024
        #---------------------------------------------------#
        self.backbone = darknet53()

        #---------------------------------------------------#
        #   out_filters : [64, 128, 256, 512, 1024]
        #---------------------------------------------------#
        out_filters = self.backbone.layers_out_filters

        #------------------------------------------------------------------------#
        #   计算yolo_head的输出通道数，对于voc数据集而言
        #   final_out_filter0 = final_out_filter1 = final_out_filter2 = 75
        #   anchors_mask[0]) * (num_classes + 5) = 3 * 25 = 75
        #------------------------------------------------------------------------#
        # 13,13,1024 -> 13,13,512 -> 13,13,75          维度变化列表  in_channels      最终conv的out_channels
        self.last_layer0            = make_last_layers([512, 1024], out_filters[-1], len(anchors_mask[0]) * (num_classes + 5))

        # 13,13,512 -> 13,13,256 -> 26,26,256
        self.last_layer1_conv       = conv2d(512, 256, 1)
        self.last_layer1_upsample   = nn.Upsample(scale_factor=2, mode='nearest')
        # 26,26,768 -> 26,26,256 -> 26,26,75
        self.last_layer1            = make_last_layers([256, 512], out_filters[-2] + 256, len(anchors_mask[1]) * (num_classes + 5))

        # 26,26,256 -> 26,26,128 -> 52,52,128
        self.last_layer2_conv       = conv2d(256, 128, 1)
        self.last_layer2_upsample   = nn.Upsample(scale_factor=2, mode='nearest')
        # 52,52,384 -> 52,52,75
        self.last_layer2            = make_last_layers([128, 256], out_filters[-3] + 128, len(anchors_mask[2]) * (num_classes + 5))

    def forward(self, x):
        #---------------------------------------------------#
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256；26,26,512；13,13,1024
        #---------------------------------------------------#
        x2, x1, x0 = self.backbone(x)

        #---------------------------------------------------#
        #   第一个特征层
        #   out0 = (b,75,13,13)
        #---------------------------------------------------#
        # 13,13,1024 -> 13,13,512
        out0_branch = self.last_layer0[:5](x0)              # 前5次特征提取,要上采样拼接
        # 13,13,512 -> 13,13,512
        out0        = self.last_layer0[5:](out0_branch)     # 后2次利用YoloHead获得预测结果

        # 13,13,512 -> 13,13,256 -> 26,26,256
        x1_in = self.last_layer1_conv(out0_branch)
        x1_in = self.last_layer1_upsample(x1_in)

        # 26,26,256 + 26,26,512 -> 26,26,768
        x1_in = torch.cat([x1_in, x1], 1)

        #---------------------------------------------------#
        #   第二个特征层
        #   out1 = (b,75,26,26)
        #---------------------------------------------------#
        # 26,26,768 -> 26,26,256
        out1_branch = self.last_layer1[:5](x1_in)
        # 26,26,256 -> 26,26,75
        out1        = self.last_layer1[5:](out1_branch)

        # 26,26,256 -> 26,26,128 -> 52,52,128
        x2_in = self.last_layer2_conv(out1_branch)
        x2_in = self.last_layer2_upsample(x2_in)

        # 52,52,128 + 52,52,256 -> 52,52,384
        x2_in = torch.cat([x2_in, x2], 1)

        #---------------------------------------------------#
        #   第三个特征层
        #   out3 = (b,75,52,52)
        #---------------------------------------------------#
        # 52,52,384 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,75
        out2 = self.last_layer2(x2_in)
        return out0, out1, out2


```

