import torch
from torch import nn, Tensor
from torch.nn import functional as F
from mobilenetv2 import mobilenetv2


class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super().__init__()
        # partial设置默认参数
        from functools import partial

        model = mobilenetv2(pretrained)
        # 不要最后的1280维度
        self.features = model.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        # 下采样8倍
        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=4))
        # 下采样16倍
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))

    def _nostride_dilate(self, m, dilate):
        """
        m: apply中的一个模型实例
        dilate: apply中的自定义参数，扩张系数
        """
        # classname = m.__class__.__name__
        # if classname.find('Conv') != -1:  # != -1 表示找得到"Conv"
        # if type(m) == nn.Conv2d:
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:4](x)
        y = self.features[4:](low_level_features)
        return low_level_features, y


if __name__ == "__main__":
    model = MobileNetV2(pretrained=False)
    x = torch.randn(1, 3, 224, 224)
    model.eval()
    y = model(x)
    print(y[0].size())  # torch.Size([1, 24, 56, 56])   下采样4倍
    print(y[1].size())  # torch.Size([1, 320, 28, 28])  下采样8倍
