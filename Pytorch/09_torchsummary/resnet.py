from torchvision import models
from torchsummary import summary


model = models.resnet18().cuda()
# summary(model, (3, 224, 224))
# ==========================================================================================
# Total params: 11,689,512
# Trainable params: 11,689,512
# Non-trainable params: 0
# Total mult-adds (G): 1.84
# ==========================================================================================
# Input size (MB): 0.57
# Forward/backward pass size (MB): 37.91
# Params size (MB): 44.59
# Estimated Total Size (MB): 83.07
# ==========================================================================================


model = models.resnet34().cuda()
# summary(model, (3, 224, 224))
# ==========================================================================================
# Total params: 21,797,672
# Trainable params: 21,797,672
# Non-trainable params: 0
# Total mult-adds (G): 3.71
# ==========================================================================================
# Input size (MB): 0.57
# Forward/backward pass size (MB): 57.05
# Params size (MB): 83.15
# Estimated Total Size (MB): 140.77
# ==========================================================================================


model = models.resnet50().cuda()
# summary(model, (3, 224, 224))
# ==========================================================================================
# Total params: 25,557,032
# Trainable params: 25,557,032
# Non-trainable params: 0
# Total mult-adds (G): 4.14
# ==========================================================================================
# Input size (MB): 0.57
# Forward/backward pass size (MB): 169.59
# Params size (MB): 97.49
# Estimated Total Size (MB): 267.66
# ==========================================================================================


model = models.resnet101().cuda()
# summary(model, (3, 224, 224))
# ==========================================================================================
# Total params: 44,549,160
# Trainable params: 44,549,160
# Non-trainable params: 0
# Total mult-adds (G): 7.89
# ==========================================================================================
# Input size (MB): 0.57
# Forward/backward pass size (MB): 247.69
# Params size (MB): 169.94
# Estimated Total Size (MB): 418.20
# ==========================================================================================


model = models.resnet152().cuda()
# summary(model, (3, 224, 224))
# ==========================================================================================
# Total params: 60,192,808
# Trainable params: 60,192,808
# Non-trainable params: 0
# Total mult-adds (G): 11.63
# ==========================================================================================
# Input size (MB): 0.57
# Forward/backward pass size (MB): 344.16
# Params size (MB): 229.62
# Estimated Total Size (MB): 574.35
# ==========================================================================================


model = models.resnext50_32x4d().cuda()
# summary(model, (3, 224, 224))
# ==========================================================================================
# Total params: 25,028,904
# Trainable params: 25,028,904
# Non-trainable params: 0
# Total mult-adds (G): 4.28
# ==========================================================================================
# Input size (MB): 0.57
# Forward/backward pass size (MB): 219.74
# Params size (MB): 95.48
# Estimated Total Size (MB): 315.79
# ==========================================================================================


model = models.resnext101_32x8d().cuda()
summary(model, (3, 224, 224))
# ==========================================================================================
# Total params: 88,791,336
# Trainable params: 88,791,336
# Non-trainable params: 0
# Total mult-adds (G): 16.59
# ==========================================================================================
# Input size (MB): 0.57
# Forward/backward pass size (MB): 476.23
# Params size (MB): 338.71
# Estimated Total Size (MB): 815.51
# ==========================================================================================