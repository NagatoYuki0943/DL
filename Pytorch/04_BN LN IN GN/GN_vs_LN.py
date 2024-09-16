import torch
from torch import nn


class GroupNorm1(nn.GroupNorm):
    """Group Normalization with 1 group, equivalent with LayerNorm.
    Input: tensor in shape [B, C, *]
    """

    def __init__(self, num_channels, **kwargs):
        #             将 num_groups 数设置为1
        super().__init__(num_groups=1, num_channels=num_channels, **kwargs)


# GN vs LN example:
B, C, H, W = 1, 3, 224, 224
x = torch.randn(B, C, H, W)

gn = nn.GroupNorm(1, C)  # 分为1组(等价LN),通道为3,数据是4维的
print(gn(x).size())  # [1, 3, 224, 224]


ln = nn.LayerNorm(
    [C, H, W]
)  # LN对于4维数据在最后3维上处理,要把 CHW 都写进参数, 对于NLP的3维,会在最后的dim维度上处理
print(ln(x).size())  # [1, 3, 224, 224]

# 实际使用LN处理图片一般会把图片的形状转换为mlp的形状 [batch, position, channel],将channel调至最后,在channel上计算LN,计算完再转换回来形状
ln = nn.LayerNorm(C)
x = x.reshape(B, C, -1).transpose(
    1, 2
)  # [1, 3, 224, 224] -> [1, 3, 224*224] -> [1, 224*224, 3]
y = ln(x)
y = y.transpose(1, 2).reshape(
    B, C, H, W
)  # [1, 224*224, 3] -> [1, 3, 224*224] -> [1, 3, 224, 224]
print(y.size())  # [1, 3, 224, 224]


# mlp序列处理实例
x = torch.randn(1, 196, 768)
ln = nn.LayerNorm(768)  # 处理最后的dim维度
print(ln(x).size())  # [1, 196, 768]
