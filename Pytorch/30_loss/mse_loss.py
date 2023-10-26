import torch
from torch import Tensor
from torch.nn import functional as F
import warnings


def mse_loss(input: Tensor, target: Tensor, reduction = 'mean'):
    """mse loss
    input and target shape should be same.

    Args:
        input (Tensor):  predict value
        target (Tensor): target value
        reduction (str, optional): mean' | 'sum' | 'none'. Defaults to 'mean'.

    Returns:
        Tensor: mse result
    """
    if target.size() != input.size():
        warnings.warn(
            "Using a target size ({}) that is different to the input size ({}). "
            "This will likely lead to incorrect results due to broadcasting. "
            "Please ensure they have the same size.".format(target.size(), input.size()),
            stacklevel=2,
        )

    result: Tensor = (input - target) ** 2
    if reduction == "mean":
        return result.mean()
    elif reduction == "sum":
        return result.sum()
    elif reduction == "none":
        return result

a = torch.tensor([
    [1, 0, 1, 0],
    [1, 1, 0, 0],
])

b = torch.tensor([
    [0.8, 0.1, 0.7, 0.3],
    [0.9, 0.6, 0.5, 0.3],
])

loss1 = F.mse_loss(a, b)
loss2 = mse_loss(a, b)
print(loss1 == loss2) # True

loss1 = F.mse_loss(a, b ,reduction = "sum")
loss2 = mse_loss(a, b, 'sum')
print(loss1 == loss2) # True

loss1 = F.mse_loss(a, b ,reduction = "none")
loss2 = mse_loss(a, b, 'none')
print(loss1 == loss2)
# True, True, True, True
# True, True, True, True
