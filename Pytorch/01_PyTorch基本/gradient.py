"""自动求导
"""

import torch


x = torch.tensor(2.0)
weight = torch.tensor(3.0, requires_grad=True)
bias = torch.tensor(4.0, requires_grad=True)
print(x.requires_grad)      # x.requires_grad = False
print(weight.requires_grad) # weight.requires_grad = True
print(bias.requires_grad)   # bias.requires_grad = True

y = weight * x + bias
print(y)            # tensor(10., grad_fn=<AddBackward0>)
print(y.grad_fn)    # <AddBackward0 object at 0x00000112A87C6AD0>

y.backward() # 反向传播,求解导数
print(x.grad)       # None
print(weight.grad)  # 2.
print(bias.grad)    # 1.


x = torch.tensor(2.0, requires_grad=True)
y = torch.log(x)
y.backward()
print(x.grad)       # 0.5000 = 1/2


x = torch.tensor(2.0, requires_grad=True)
y = x ** 3
y.backward()
print(x.grad)       # 12. = 3 * 2^2
