import torch
from torch import nn
from copy import copy, deepcopy

torch.manual_seed(0)

# ------------------------------------#
#   需要使用 deepcopy 复制一份新模型
# ------------------------------------#

model1 = nn.Linear(1, 1)
model2 = model1
model3 = copy(model1)
model4 = deepcopy(model1)  # deepcopy 可以完全复制一份模型

print(
    model1.state_dict()
)  # OrderedDict([('weight', tensor([[-0.0075]])), ('bias', tensor([0.5364]))])
print(
    model2.state_dict()
)  # OrderedDict([('weight', tensor([[-0.0075]])), ('bias', tensor([0.5364]))])
print(
    model3.state_dict()
)  # OrderedDict([('weight', tensor([[-0.0075]])), ('bias', tensor([0.5364]))])
print(
    model4.state_dict()
)  # OrderedDict([('weight', tensor([[-0.0075]])), ('bias', tensor([0.5364]))])
print()

# 更新 model1 参数
for parameter in model1.parameters():
    parameter.data = parameter.data * 10

print(
    model1.state_dict()
)  # OrderedDict([('weight', tensor([[-0.0749]])), ('bias', tensor([5.3644]))])
print(
    model2.state_dict()
)  # OrderedDict([('weight', tensor([[-0.0749]])), ('bias', tensor([5.3644]))]) ❌ =        修改
print(
    model3.state_dict()
)  # OrderedDict([('weight', tensor([[-0.0749]])), ('bias', tensor([5.3644]))]) ❌ copy     修改
print(
    model4.state_dict()
)  # OrderedDict([('weight', tensor([[-0.0075]])), ('bias', tensor([0.5364]))]) ✔️ deepcopy 没有修改
