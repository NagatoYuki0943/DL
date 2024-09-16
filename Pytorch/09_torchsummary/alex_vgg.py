from torchvision import models
from torchsummary import summary


model = models.alexnet().cuda()
# summary(model, (3, 224, 224))
# ==========================================================================================
# Total params: 61,100,840
# Trainable params: 61,100,840
# Non-trainable params: 0
# Total mult-adds (M): 775.28
# ==========================================================================================
# Input size (MB): 0.57
# Forward/backward pass size (MB): 3.77
# Params size (MB): 233.08
# Estimated Total Size (MB): 237.43
# ==========================================================================================


model = models.vgg11().cuda()
# summary(model, (3, 224, 224))
# ==========================================================================================
# Total params: 132,863,336
# Trainable params: 132,863,336
# Non-trainable params: 0
# Total mult-adds (G): 7.74
# ==========================================================================================
# Input size (MB): 0.57
# Forward/backward pass size (MB): 56.73
# Params size (MB): 506.83
# Estimated Total Size (MB): 564.13
# ==========================================================================================


model = models.vgg11_bn().cuda()
# summary(model, (3, 224, 224))
# ==========================================================================================
# Total params: 132,868,840
# Trainable params: 132,868,840
# Non-trainable params: 0
# Total mult-adds (G): 7.74
# ==========================================================================================
# Input size (MB): 0.57
# Forward/backward pass size (MB): 113.38
# Params size (MB): 506.85
# Estimated Total Size (MB): 620.81
# ==========================================================================================


model = models.vgg13().cuda()
# summary(model, (3, 224, 224))
# ==========================================================================================
# Total params: 133,047,848
# Trainable params: 133,047,848
# Non-trainable params: 0
# Total mult-adds (G): 11.44
# ==========================================================================================
# Input size (MB): 0.57
# Forward/backward pass size (MB): 93.48
# Params size (MB): 507.54
# Estimated Total Size (MB): 601.59
# ==========================================================================================


model = models.vgg13_bn().cuda()
# summary(model, (3, 224, 224))
# ==========================================================================================
# Total params: 133,053,736
# Trainable params: 133,053,736
# Non-trainable params: 0
# Total mult-adds (G): 11.44
# ==========================================================================================
# Input size (MB): 0.57
# Forward/backward pass size (MB): 186.88
# Params size (MB): 507.56
# Estimated Total Size (MB): 695.02
# ==========================================================================================


model = models.vgg16().cuda()
# summary(model, (3, 224, 224))
# ==========================================================================================
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0
# Total mult-adds (G): 15.61
# ==========================================================================================
# Input size (MB): 0.57
# Forward/backward pass size (MB): 103.43
# Params size (MB): 527.79
# Estimated Total Size (MB): 631.80
# ==========================================================================================


model = models.vgg16_bn().cuda()
# summary(model, (3, 224, 224))
# ==========================================================================================
# Total params: 138,365,992
# Trainable params: 138,365,992
# Non-trainable params: 0
# Total mult-adds (G): 15.61
# ==========================================================================================
# Input size (MB): 0.57
# Forward/backward pass size (MB): 206.79
# Params size (MB): 527.82
# Estimated Total Size (MB): 735.19
# ==========================================================================================


model = models.vgg19().cuda()
# summary(model, (3, 224, 224))
# ==========================================================================================
# Total params: 143,667,240
# Trainable params: 143,667,240
# Non-trainable params: 0
# Total mult-adds (G): 19.78
# ==========================================================================================
# Input size (MB): 0.57
# Forward/backward pass size (MB): 113.38
# Params size (MB): 548.05
# Estimated Total Size (MB): 662.00
# ==========================================================================================


model = models.vgg19_bn().cuda()
# summary(model, (3, 224, 224))
# ==========================================================================================
# Total params: 143,678,248
# Trainable params: 143,678,248
# Non-trainable params: 0
# Total mult-adds (G): 19.78
# ==========================================================================================
# Input size (MB): 0.57
# Forward/backward pass size (MB): 226.70
# Params size (MB): 548.09
# Estimated Total Size (MB): 775.36
# ==========================================================================================
