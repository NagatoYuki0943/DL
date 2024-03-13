"""自动求导
"""

import torch


x = torch.tensor(2.0, requires_grad=True)
y = torch.log(x)
y.backward()
print(x.grad)       # 0.5000 = 1/2


x = torch.tensor(2.0, requires_grad=True)
y = x ** 3
y.backward()
print(x.grad)       # 12. = 3 * 2^2

print("=" * 100)

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

print("=" * 100)

#------------------------------------------------#
#   x1   weight1  bias1     x2   weight2  bias2
#   │       │       │       │       │       │
#   └── * ──┘       │       └── * ──┘       │
#       │           │           │           │
#       └──── + ────┘           └──── + ────┘
#             │                       │
#  weight3    y1   bias3   weight4    y2
#     │       │      │        │       │
#     └── * ──┘      │        └── * ──┘
#         │          │            │
#         └───────── + ───────────┘
#                    │
#                    z
#------------------------------------------------#
x1 = torch.tensor(2.0)
weight1 = torch.tensor(3.0, requires_grad=True)
bias1 = torch.tensor(4.0, requires_grad=True)
y1 = weight1 * x1 + bias1   # 3 * 2 + 4 = 10

x2 = torch.tensor(4.0)
weight2 = torch.tensor(5.0, requires_grad=True)
bias2 = torch.tensor(6.0, requires_grad=True)
y2 = weight2 * x2 + bias2   # 5 * 4 + 6 = 26

weight3 = torch.tensor(5.0, requires_grad=True)
weight4 = torch.tensor(6.0, requires_grad=True)
bias3 = torch.tensor(7.0, requires_grad=True)

z = weight3 * y1 + weight4 * y2 + bias3    # 5 * 10 + 6 * 26 + 7 = 213
z.backward()
print(weight3.grad) # 10 = 2 * 3 + 4
print(weight4.grad) # 26 = 4 * 5 + 6
print(bias3.grad)   # 1

print(weight1.grad) # d(z/weight1) = d(z/y1) * d(y1/weight1) = 5 * 2 = 10
print(bias1.grad)   # d(z/bias1)   = d(z/y1) * d(y1/bias1)   = 5 * 1 = 5
print(weight2.grad) # d(z/weight2) = d(z/y2) * d(y2/weight2) = 6 * 4 = 24
print(bias2.grad)   # d(z/bias2)   = d(z/y2) * d(y2/bias2)   = 6 * 1 = 6

print("=" * 100)

#------------------------------------------------#
#                       ┌ x ┐
#     bias1  weight1    │   │   weight2  bias2
#       │       │       │   │       │       │
#       │       └── * ──┘   └── * ──┘       │
#       │           │           │           │
#       └──── + ────┘           └──── + ────┘
#             │                       │
#  weight3    y1   bias3   weight4    y2
#     │       │      │        │       │
#     └── * ──┘      │        └── * ──┘
#         │          │            │
#         └───────── + ───────────┘
#                    │
#                    z
#------------------------------------------------#
x = torch.tensor(3.0, requires_grad=True)
weight1 = torch.tensor(2.0, requires_grad=True)
bias1 = torch.tensor(1.0, requires_grad=True)
y1 = weight1 * x + bias1

weight2 = torch.tensor(3.0, requires_grad=True)
bias2 = torch.tensor(2.0, requires_grad=True)
y2 = weight2 * x + bias2

weight3 = torch.tensor(4.0, requires_grad=True)
weight4 = torch.tensor(5.0, requires_grad=True)
bias3 = torch.tensor(6.0, requires_grad=True)
z = y1 * weight3 + y2 * weight4 + bias3

z.backward()

print(x.grad)   # d(z/x) = d(z/y1) * d(y1/x) + d(z/y2) * d(y2/x) = 4 * 2 + 5 * 3 = 23
