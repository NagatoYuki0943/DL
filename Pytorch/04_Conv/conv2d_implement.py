import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F


torch.manual_seed(0)


def conv2d(x: Tensor, weight: Tensor, bias: Tensor | None = None):
    """
    Compute a 2D convolution on the input tensor `x` using the weight tensor `weight` and an optional bias tensor `bias`.

    Parameters:
        x (Tensor): The input tensor of shape (height, width).
        weight (Tensor): The weight tensor of shape (kernel_height, kernel_width).
        bias (Tensor, optional): An optional bias tensor of shape (1,). Defaults to None.

    Returns:
        Tensor: The output tensor of shape (height - kernel_height + 1, width - kernel_width + 1).
    """

    assert len(x.shape) == 2
    assert len(weight.shape) == 2

    h, w = weight.shape
    y = torch.zeros((x.shape[0] - h + 1, x.shape[1] - w + 1))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i, j] = (x[i : i + h, j : j + w] * weight).sum()
    if bias:
        y += bias
    return y


height, width = 5, 5
kernel_height, kernel_width = 3, 3
x = torch.ones(height, width)
weight = torch.randn(kernel_height, kernel_width)
bias = torch.randn(1)

conv1 = nn.Conv2d(1, 1, (kernel_height, kernel_width), stride=1, padding=0, bias=True)
print(conv1.weight.data.shape)  # [1, 1, 3, 3]
print(conv1.bias.data.shape)  # [1]

conv1.weight.data[:] = weight.reshape(1, 1, kernel_height, kernel_width)
conv1.bias.data[:] = bias

print(weight.sum() + bias)
# [-2.7271]

y1 = conv2d(x, weight, bias)
print(y1)
# [[-2.7271, -2.7271, -2.7271],
#  [-2.7271, -2.7271, -2.7271],
#  [-2.7271, -2.7271, -2.7271]]

with torch.inference_mode():
    y2 = conv1(x.reshape(1, 1, height, width)).squeeze()
    print(y2)
    # [[-2.7271, -2.7271, -2.7271],
    #  [-2.7271, -2.7271, -2.7271],
    #  [-2.7271, -2.7271, -2.7271]]
    y3 = F.conv2d(
        x.reshape(1, 1, height, width),
        weight.reshape(1, 1, kernel_height, kernel_width),
        bias,
    ).squeeze()
    print(y3)
    # [[-2.7271, -2.7271, -2.7271],
    #  [-2.7271, -2.7271, -2.7271],
    #  [-2.7271, -2.7271, -2.7271]]

print(torch.allclose(y1, y2), torch.allclose(y1, y3), torch.all(y2 == y3))
# True True tensor(True)
