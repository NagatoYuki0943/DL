import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F


torch.manual_seed(0)


def conv3d(x: Tensor, weight: Tensor, bias: Tensor | None = None):
    """
    Compute a 3D convolution on the input tensor `x` using the weight tensor `weight` and an optional bias tensor `bias`.

    Parameters:
        x (Tensor): The input tensor of shape (depth, height, width).
        weight (Tensor): The weight tensor of shape (kernel_depth, kernel_height, kernel_width).
        bias (Tensor, optional): An optional bias tensor of shape (1,). Defaults to None.

    Returns:
        Tensor: The output tensor of shape (depth - kernel_depth + 1, height - kernel_height + 1, width - kernel_width + 1).
    """

    assert len(x.shape) == 3
    assert len(weight.shape) == 3

    d, h, w = weight.shape
    y = torch.zeros((x.shape[0] - d + 1, x.shape[1] - h + 1, x.shape[2] - w + 1))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            for k in range(y.shape[2]):
                y[i, j, k] = (x[i:i + d, j:j + h, k:k + w] * weight).sum()
    if bias:
        y += bias
    return y


depth, height, width = 5, 5, 5
kernel_depth, kernel_height, kernel_width = 3, 3, 3
x = torch.ones(depth, height, width)
weight = torch.randn(kernel_depth, kernel_height, kernel_width)
bias = torch.randn(1)

conv1 = nn.Conv3d(1, 1, (kernel_depth, kernel_height, kernel_width), stride=1, padding=0, bias=True)
print(conv1.weight.data.shape)  # [1, 1, 3, 3, 3]
print(conv1.bias.data.shape)    # [1]

conv1.weight.data[:] = weight.reshape(1, 1, kernel_depth, kernel_height, kernel_width)
conv1.bias.data[:] = bias

print(weight.sum() + bias)
# [-7.1899]

y1 = conv3d(x, weight, bias)
print(y1)
# [[[-7.1899, -7.1899, -7.1899],
#   [-7.1899, -7.1899, -7.1899],
#   [-7.1899, -7.1899, -7.1899]],
#  [[-7.1899, -7.1899, -7.1899],
#   [-7.1899, -7.1899, -7.1899],
#   [-7.1899, -7.1899, -7.1899]],
#  [[-7.1899, -7.1899, -7.1899],
#   [-7.1899, -7.1899, -7.1899],
#   [-7.1899, -7.1899, -7.1899]]]

with torch.inference_mode():
    y2 = conv1(x.reshape(1, 1, depth, height, width)).squeeze()
    print(y2)
    # [[[-7.1899, -7.1899, -7.1899],
    #   [-7.1899, -7.1899, -7.1899],
    #   [-7.1899, -7.1899, -7.1899]],
    #  [[-7.1899, -7.1899, -7.1899],
    #   [-7.1899, -7.1899, -7.1899],
    #   [-7.1899, -7.1899, -7.1899]],
    #  [[-7.1899, -7.1899, -7.1899],
    #   [-7.1899, -7.1899, -7.1899],
    #   [-7.1899, -7.1899, -7.1899]]]
    y3 = F.conv3d(x.reshape(1, 1, depth, height, width), weight.reshape(1, 1, kernel_depth, kernel_height, kernel_width), bias).squeeze()
    print(y3)
    # [[[-7.1899, -7.1899, -7.1899],
    #   [-7.1899, -7.1899, -7.1899],
    #   [-7.1899, -7.1899, -7.1899]],
    #  [[-7.1899, -7.1899, -7.1899],
    #   [-7.1899, -7.1899, -7.1899],
    #   [-7.1899, -7.1899, -7.1899]],
    #  [[-7.1899, -7.1899, -7.1899],
    #   [-7.1899, -7.1899, -7.1899],
    #   [-7.1899, -7.1899, -7.1899]]]

print(torch.allclose(y1, y2), torch.allclose(y1, y3), torch.all(y2 == y3))
# True True tensor(True)
