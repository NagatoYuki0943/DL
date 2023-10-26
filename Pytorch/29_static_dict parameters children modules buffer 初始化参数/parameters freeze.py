import torch
from torchvision.models import vgg16
from torch.optim import Adam

model = vgg16()


"""
可以通过named_parameters冻结权重
"""
for k, v in model.named_parameters():
    if 'features' in k:
        v.requires_grad = False
    else:
        print(f"training: {k}")
# 只要训练的参数
pg = [p for p in model.parameters() if p.requires_grad]
optimizer = Adam(pg, lr=0.001)