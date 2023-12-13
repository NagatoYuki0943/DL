import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F


torch.manual_seed(0)


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(out_features, in_features))
        self.bias = nn.Parameter(torch.rand(out_features)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        x = x @ self.weight.T
        if self.bias is not None:
            x += self.bias
        return x


x = torch.randn(1, 10)
linear1 = Linear(10, 5, bias=True).eval()
linear2 = nn.Linear(10, 5, bias=True).eval()

print(linear1.weight.data.shape)# [5, 10]
print(linear1.bias.data.shape)  # [5]
print(linear2.weight.data.shape)# [5, 10]
print(linear2.bias.data.shape)  # [5]

# repleace weight and bias
linear2.weight.data[:] = linear1.weight.data
linear2.bias.data[:] = linear1.bias.data

with torch.inference_mode():
    print(linear1(x))                               # [[-0.5692,  0.1194,  0.2910, -0.3820,  0.0709]]
    print(linear2(x))                               # [[-0.5692,  0.1194,  0.2910, -0.3820,  0.0709]]
    print(F.linear(x, linear1.weight, linear1.bias))# [[-0.5692,  0.1194,  0.2910, -0.3820,  0.0709]]
