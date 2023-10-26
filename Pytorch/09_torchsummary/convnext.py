from torchvision import models
from torchsummary import summary


model = models.convnext_tiny().cuda()
# summary(model, (3, 224, 224))
# ===============================================================================================
# Total params: 28,589,128
# Trainable params: 28,589,128
# Non-trainable params: 0
# Total mult-adds (M): 297.34
# ===============================================================================================
# Input size (MB): 0.57
# Forward/backward pass size (MB): 10.64
# Params size (MB): 109.06
# Estimated Total Size (MB): 120.27
# ===============================================================================================


model = models.convnext_small().cuda()
# summary(model, (3, 224, 224))
# ===============================================================================================
# Total params: 50,223,688
# Trainable params: 50,223,688
# Non-trainable params: 0
# Total mult-adds (M): 383.66
# ===============================================================================================
# Input size (MB): 0.57
# Forward/backward pass size (MB): 10.64
# Params size (MB): 191.59
# Estimated Total Size (MB): 202.80
# ===============================================================================================


model = models.convnext_base().cuda()
# summary(model, (3, 224, 224))
# ===============================================================================================
# Total params: 88,591,464
# Trainable params: 88,591,464
# Non-trainable params: 0
# Total mult-adds (M): 673.75
# ===============================================================================================
# Input size (MB): 0.57
# Forward/backward pass size (MB): 14.18
# Params size (MB): 337.95
# Estimated Total Size (MB): 352.70
# ===============================================================================================


model = models.convnext_large().cuda()
# summary(model, (3, 224, 224))
# ===============================================================================================
# Total params: 197,767,336
# Trainable params: 197,767,336
# Non-trainable params: 0
# Total mult-adds (G): 1.50
# ===============================================================================================
# Input size (MB): 0.57
# Forward/backward pass size (MB): 21.27
# Params size (MB): 754.42
# Estimated Total Size (MB): 776.26
# ===============================================================================================