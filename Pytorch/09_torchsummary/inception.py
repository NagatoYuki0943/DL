from torchvision import models
from torchsummary import summary
from models import inceptionv4, inceptionresnetv2, xception


model = models.googlenet(init_weights = True)
# summary(model, (3, 299, 299))
# ==========================================================================================
# Total params: 6,624,904
# Trainable params: 6,624,904
# Non-trainable params: 0
# Total mult-adds (G): 2.60
# ==========================================================================================
# Input size (MB): 1.02
# Forward/backward pass size (MB): 85.95
# Params size (MB): 25.27
# Estimated Total Size (MB): 112.25
# ==========================================================================================


model = models.inception_v3(init_weights = True)
# summary(model, (3, 299, 299))
# ==========================================================================================
# Total params: 23,834,568
# Trainable params: 23,834,568
# Non-trainable params: 0
# Total mult-adds (G): 5.76
# ==========================================================================================
# Input size (MB): 1.02
# Forward/backward pass size (MB): 136.84
# Params size (MB): 90.92
# Estimated Total Size (MB): 228.79
# ==========================================================================================



model = inceptionv4(pretrained=False)
# summary(model, (3, 224, 224))
# ==========================================================================================
# Total params: 42,679,816
# Trainable params: 42,679,816
# Non-trainable params: 0
# Total mult-adds (G): 1.85
# ==========================================================================================
# Input size (MB): 0.57
# Forward/backward pass size (MB): 43.38
# Params size (MB): 162.81
# Estimated Total Size (MB): 206.77
# ==========================================================================================


model = inceptionresnetv2(pretrained=False)
# summary(model, (3, 299, 299))
# ==========================================================================================
# Total params: 55,843,464
# Trainable params: 55,843,464
# Non-trainable params: 0
# Total mult-adds (G): 9.16
# ==========================================================================================
# Input size (MB): 1.02
# Forward/backward pass size (MB): 207.22
# Params size (MB): 213.03
# Estimated Total Size (MB): 421.26
# ==========================================================================================


model = xception(pretrained=False)
# summary(model, (3, 224, 224))
# ==========================================================================================
# Total params: 22,855,952
# Trainable params: 22,855,952
# Non-trainable params: 0
# Total mult-adds (G): 4.60
# ==========================================================================================
# Input size (MB): 0.57
# Forward/backward pass size (MB): 238.59
# Params size (MB): 87.19
# Estimated Total Size (MB): 326.35
# ==========================================================================================