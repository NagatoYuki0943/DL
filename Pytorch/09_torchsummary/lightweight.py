from torchvision import models
from torchsummary import summary


model = models.squeezenet1_0()
# summary(model, (3, 224, 224))
# ==========================================================================================
# Total params: 1,248,424
# Trainable params: 1,248,424
# Non-trainable params: 0
# Total mult-adds (M): 820.89
# ==========================================================================================
# Input size (MB): 0.57
# Forward/backward pass size (MB): 33.16
# Params size (MB): 4.76
# Estimated Total Size (MB): 38.50
# ==========================================================================================


model = models.squeezenet1_1()
# summary(model, (3, 224, 224))
# ==========================================================================================
# Total params: 1,235,496
# Trainable params: 1,235,496
# Non-trainable params: 0
# Total mult-adds (M): 351.10
# ==========================================================================================
# Input size (MB): 0.57
# Forward/backward pass size (MB): 19.76
# Params size (MB): 4.71
# Estimated Total Size (MB): 25.04
# ==========================================================================================



model = models.shufflenet_v2_x0_5()
# summary(model, (3, 224, 224))
# ==========================================================================================
# Total params: 1,366,792
# Trainable params: 1,366,792
# Non-trainable params: 0
# Total mult-adds (M): 41.10
# ==========================================================================================
# Input size (MB): 0.57
# Forward/backward pass size (MB): 15.63
# Params size (MB): 5.21
# Estimated Total Size (MB): 21.42
# ==========================================================================================


model = models.shufflenet_v2_x1_0()
# summary(model, (3, 224, 224))
# ==========================================================================================
# Total params: 2,278,604
# Trainable params: 2,278,604
# Non-trainable params: 0
# Total mult-adds (M): 147.70
# ==========================================================================================
# Input size (MB): 0.57
# Forward/backward pass size (MB): 29.77
# Params size (MB): 8.69
# Estimated Total Size (MB): 39.03
# ==========================================================================================


model = models.shufflenet_v2_x1_5()
# summary(model, (3, 224, 224))
# ==========================================================================================
# Total params: 3,503,624
# Trainable params: 3,503,624
# Non-trainable params: 0
# Total mult-adds (M): 301.73
# ==========================================================================================
# Input size (MB): 0.57
# Forward/backward pass size (MB): 42.24
# Params size (MB): 13.37
# Estimated Total Size (MB): 56.18
# ==========================================================================================


model = models.shufflenet_v2_x2_0()
# summary(model, (3, 224, 224))
# ==========================================================================================
# Total params: 7,393,996
# Trainable params: 7,393,996
# Non-trainable params: 0
# Total mult-adds (M): 595.25
# ==========================================================================================
# Input size (MB): 0.57
# Forward/backward pass size (MB): 57.14
# Params size (MB): 28.21
# Estimated Total Size (MB): 85.92
# ==========================================================================================


model = model = models.mobilenet_v2()
summary(model, (3, 224, 224))
# ===============================================================================================
# Total params: 3,504,872
# Trainable params: 3,504,872
# Non-trainable params: 0
# Total mult-adds (M): 158.62
# ===============================================================================================
# Input size (MB): 0.57
# Forward/backward pass size (MB): 15.82
# Params size (MB): 13.37
# Estimated Total Size (MB): 29.77
# ===============================================================================================


model = model = models.mobilenet_v3_small()
# summary(model, (3, 224, 224))
# ===============================================================================================
# Total params: 2,542,856
# Trainable params: 2,542,856
# Non-trainable params: 0
# Total mult-adds (M): 14.92
# ===============================================================================================
# Input size (MB): 0.57
# Forward/backward pass size (MB): 3.51
# Params size (MB): 9.70
# Estimated Total Size (MB): 13.78
# ===============================================================================================


model = model = models.mobilenet_v3_large()
# summary(model, (3, 224, 224))
# ===============================================================================================
# Total params: 5,483,032
# Trainable params: 5,483,032
# Non-trainable params: 0
# Total mult-adds (M): 29.47
# ===============================================================================================
# Input size (MB): 0.57
# Forward/backward pass size (MB): 3.80
# Params size (MB): 20.92
# Estimated Total Size (MB): 25.29
# ===============================================================================================
