import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.hub import load_state_dict_from_url
import math
import os


def conv_bn(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(True),
    )


def conv_1x1_bn(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(True),
    )


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_channels = round(in_channels * expand_ratio)
        # 步长为1同时进出通道相同
        self.use_res_connect = stride == 1 and in_channels == out_channels

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # --------------------------------------------#
                #   进行3x3的逐层卷积，进行跨特征点的特征提取
                # --------------------------------------------#
                nn.Conv2d(
                    hidden_channels,
                    hidden_channels,
                    3,
                    stride,
                    1,
                    groups=hidden_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU6(True),
                # -----------------------------------#
                #   利用1x1卷积进行通道数的调整
                # -----------------------------------#
                nn.Conv2d(hidden_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.conv = nn.Sequential(
                # -----------------------------------#
                #   利用1x1卷积进行通道数的上升
                # -----------------------------------#
                nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU6(True),
                # --------------------------------------------#
                #   进行3x3的逐层卷积，进行跨特征点的特征提取
                # --------------------------------------------#
                nn.Conv2d(
                    hidden_channels,
                    hidden_channels,
                    3,
                    stride,
                    1,
                    groups=hidden_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU6(True),
                # -----------------------------------#
                #   利用1x1卷积进行通道数的调整
                # -----------------------------------#
                nn.Conv2d(hidden_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, input_size=224, width_mult=1.0):
        super().__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # 扩展倍率， 输出通道数， 重复次数， 步长
            # t, c, n, s
            [1, 16, 1, 1],  # 256,256, 32 -> 256, 256,16
            [6, 24, 2, 2],  # 256,256, 16 -> 128, 128,24   2
            [6, 32, 3, 2],  # 128,128, 24 ->  64, 64, 32   4
            [6, 64, 4, 2],  #  64, 64, 32 ->  32, 32, 64   7
            [6, 96, 3, 1],  #  32, 32, 64 ->  32, 32, 96
            [6, 160, 3, 2],  #  32, 32, 96 ->  16, 16,160   14
            [6, 320, 1, 1],  #  16, 16,160 ->  16, 16,320
        ]

        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = (
            int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        )

        # 512, 512, 3 -> 256, 256, 32
        features = [conv_bn(3, input_channel, 2)]

        # 扩展倍率， 输出通道数， 重复次数， 步长
        for t, c, n, s in interverted_residual_setting:
            out_channels = int(c * width_mult)
            for i in range(n):
                # 第一次下采样，其余步长为1
                if i == 0:
                    features.append(
                        block(input_channel, out_channels, stride=s, expand_ratio=t)
                    )
                else:
                    features.append(
                        block(input_channel, out_channels, stride=1, expand_ratio=t)
                    )
                input_channel = out_channels

        features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features = nn.Sequential(*features)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(self.last_channel, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        print(x.size())  # [1, 1280, 7, 7]
        # x = x.mean(3).mean(2)
        x = self.pool(x)
        print(x.size())  # [1, 1280, 1, 1]
        # x = x.view(x.size(0), -1)
        x = torch.flatten(x, 1)
        print(x.size())  # [1, 1280]
        return self.classifier(x)


def load_url(url, model_dir="./model_data", map_location=None):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    filename = url.split("/")[-1]
    cached_file = os.path.join(model_dir, filename)
    if os.path.exists(cached_file):
        return torch.load(cached_file, map_location)
    else:
        return load_state_dict_from_url(url, model_dir=model_dir)


def mobilenetv2(pretrained=False, **kwargs):
    model = MobileNetV2(num_classes=1000, **kwargs)
    # if pretrained:
    #    model.load_state_dict(load_url('https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/mobilenet_v2.pth.tar'), strict=False)
    return model


if __name__ == "__main__":
    model = mobilenetv2(pretrained=True)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.size())  # [1, 1000]
    print(type(model))
    print(isinstance(model, MobileNetV2))  # True
    print(type(model) == MobileNetV2)  # True
