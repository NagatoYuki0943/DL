import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F


torch.manual_seed(0)


def conv1d(x: Tensor, weight: Tensor, bias: Tensor | None = None):
    """
    Compute a 1D convolution on the input tensor `x` using the weight tensor `weight` and an optional bias tensor `bias`.

    Parameters:
        x (Tensor): The input tensor of shape (length).
        weight (Tensor): The weight tensor of shape (kernel_length).
        bias (Tensor, optional): An optional bias tensor of shape (1,). Defaults to None.

    Returns:
        Tensor: The output tensor of shape (length - kernel_length + 1).
    """

    assert len(x.shape) == 1
    assert len(weight.shape) == 1

    l = weight.shape[0]
    y = torch.zeros(x.shape[0] - l + 1)
    for i in range(y.shape[0]):
        y[i] = (x[i:i + l] * weight).sum()
    if bias is not None:
        y += bias
    return y


length = 5
kernel_length = 3
x = torch.ones(length)
weight = torch.randn(kernel_length)
bias = torch.randn(1)

conv1 = nn.Conv1d(1, 1, kernel_length, stride=1, padding=0, bias=True)
print(conv1.weight.data.shape)  # [1, 3, 3]
print(conv1.bias.data.shape)    # [1]

conv1.weight.data[:] = weight.reshape(1, 1, kernel_length)
conv1.bias.data[:] = bias

print(weight.sum() + bias)
# [-0.3628]

y1 = conv1d(x, weight, bias)
print(y1)
# [-0.3628, -0.3628, -0.3628]

with torch.inference_mode():
    y2 = conv1(x.reshape(1, 1, length)).squeeze()
    print(y2)
    # [-0.3628, -0.3628, -0.3628]
    y3 = F.conv1d(x.reshape(1, 1, length), weight.reshape(1, 1, kernel_length), bias).squeeze()
    print(y3)
    # [-0.3628, -0.3628, -0.3628]

print(torch.allclose(y1, y2), torch.allclose(y1, y3), torch.all(y2 == y3))
# True True tensor(True)
