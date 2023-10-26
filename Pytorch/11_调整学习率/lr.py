import torch
from torch import optim
from torchvision import models
from torchvision.models import resnet18


model = resnet18()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# StepLR 按照阶段衰减
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
# MultiStepLR 按阶段衰减
#scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40], gamma=0.1)
# 余弦退火
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)

def train():
    optimizer.zero_grad()

    print(optimizer.state_dict()['param_groups'][0]['lr'])
    optimizer.step()

    scheduler.step()


for epoch in range(50):
    train()