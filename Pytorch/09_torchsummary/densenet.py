from torchvision import models
from torchsummary import summary


model = models.densenet121().cuda()
# summary(model, (3, 224, 224))
# ==========================================================================================
# Total params: 7,978,856
# Trainable params: 7,978,856
# Non-trainable params: 0
# Total mult-adds (G): 2.85
# ==========================================================================================
# Input size (MB): 0.57
# Forward/backward pass size (MB): 172.18
# Params size (MB): 30.44
# Estimated Total Size (MB): 203.19
# ==========================================================================================


model = models.densenet169().cuda()
# summary(model, (3, 224, 224))
# ==========================================================================================
# Total params: 14,149,480
# Trainable params: 14,149,480
# Non-trainable params: 0
# Total mult-adds (G): 3.40
# ==========================================================================================
# Input size (MB): 0.57
# Forward/backward pass size (MB): 200.84
# Params size (MB): 53.98
# Estimated Total Size (MB): 255.39
# ==========================================================================================


model = models.densenet201().cuda()
# summary(model, (3, 224, 224))
# ==========================================================================================
# Total params: 20,013,928
# Trainable params: 20,013,928
# Non-trainable params: 0
# Total mult-adds (G): 4.34
# ==========================================================================================
# Input size (MB): 0.57
# Forward/backward pass size (MB): 248.41
# Params size (MB): 76.35
# Estimated Total Size (MB): 325.33
# ==========================================================================================


model = models.densenet161().cuda()
# summary(model, (3, 224, 224))
# ==========================================================================================
# Total params: 28,681,000
# Trainable params: 28,681,000
# Non-trainable params: 0
# Total mult-adds (G): 7.80
# ==========================================================================================
# Input size (MB): 0.57
# Forward/backward pass size (MB): 308.83
# Params size (MB): 109.41
# Estimated Total Size (MB): 418.81
# ==========================================================================================