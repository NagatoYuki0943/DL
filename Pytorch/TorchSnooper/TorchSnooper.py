import torch
from torch import nn, Tensor
import torchsnooper


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 64, 3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        return x


@torchsnooper.snoop()
def test():
    x = torch.ones(1, 3, 224, 224)
    model = Net().cuda()  # 将模型放到显卡上
    model.eval()
    with torch.inference_mode():
        y = model(x)
    print(y.size())


test()
