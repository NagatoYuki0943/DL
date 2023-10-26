"""
预训练权重和模型都能用 state_dict.items() 来获取 k 和 v,
k可以修改,满足迁移学习的需要
"""


import torch
from torchvision.models import vgg16


pre_state_dict = torch.load(f"D:\AI\预训练权重\\vgg16-397923af.pth")
for k, v in pre_state_dict.items():
    print(k)
    # features.0.weight
    # features.0.bias
    # features.2.weight
    # features.2.bias
    # features.5.weight
    # features.5.bias
    # features.7.weight
    # features.7.bias
    # features.10.weight
    # features.10.bias
    # features.12.weight
    # features.12.bias
    # features.14.weight
    # features.14.bias
    # features.17.weight
    # features.17.bias
    # features.19.weight
    # features.19.bias
    # features.21.weight
    # features.21.bias
    # features.24.weight
    # features.24.bias
    # features.26.weight
    # features.26.bias
    # features.28.weight
    # features.28.bias
    # classifier.0.weight
    # classifier.0.bias
    # classifier.3.weight
    # classifier.3.bias
    # classifier.6.weight
    # classifier.6.bias

print('*'*100)
model = vgg16()
state_dict = model.state_dict()
for k, v in state_dict.items():
    print(k)
    # features.0.weight
    # features.0.bias
    # features.2.weight
    # features.2.bias
    # features.5.weight
    # features.5.bias
    # features.7.weight
    # features.7.bias
    # features.10.weight
    # features.10.bias
    # features.12.weight
    # features.12.bias
    # features.14.weight
    # features.14.bias
    # features.17.weight
    # features.17.bias
    # features.19.weight
    # features.19.bias
    # features.21.weight
    # features.21.bias
    # features.24.weight
    # features.24.bias
    # features.26.weight
    # features.26.bias
    # features.28.weight
    # features.28.bias
    # classifier.0.weight
    # classifier.0.bias
    # classifier.3.weight
    # classifier.3.bias
    # classifier.6.weight
    # classifier.6.bias