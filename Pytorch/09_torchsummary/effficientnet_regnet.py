from torchvision import models
from torchsummary import summary
from models import efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l

#------------------------------------#
# efficientnet输入图像分辨率大小
# b0 224x224
# b1 240x240
# b2 260x260
# b3 300x300
# b4 380x380
# b5 456x456
# b6 528x528
# b7 600x600
#
# efficientnetv2
# 对于S:
#   train_size = 300
#   valid_size = 384
# 对于M:
#   train_size = 384
#   valid_size = 480
# 对于L:
#   train_size = 384
#   valid_size = 480
#------------------------------------#


model = models.efficientnet_b0().cuda()
# summary(model, (3, 224, 224))
# ===============================================================================================
# Total params: 5,288,548
# Trainable params: 5,288,548
# Non-trainable params: 0
# Total mult-adds (M): 48.55
# ===============================================================================================
# Input size (MB): 0.57
# Forward/backward pass size (MB): 7.09
# Params size (MB): 20.17
# Estimated Total Size (MB): 27.84
# ===============================================================================================


model = models.efficientnet_b1().cuda()
# summary(model, (3, 240, 240))
# ===============================================================================================
# Total params: 7,794,184
# Trainable params: 7,794,184
# Non-trainable params: 0
# Total mult-adds (M): 66.27
# ===============================================================================================
# Input size (MB): 0.66
# Forward/backward pass size (MB): 8.29
# Params size (MB): 29.73
# Estimated Total Size (MB): 38.68
# ===============================================================================================


model = models.efficientnet_b2().cuda()
# summary(model, (3, 260, 260))
# ===============================================================================================
# Total params: 9,109,994
# Trainable params: 9,109,994
# Non-trainable params: 0
# Total mult-adds (M): 87.18
# ===============================================================================================
# Input size (MB): 0.77
# Forward/backward pass size (MB): 10.00
# Params size (MB): 34.75
# Estimated Total Size (MB): 45.53
# ===============================================================================================


model = models.efficientnet_b3().cuda()
# summary(model, (3, 300, 300))
# ===============================================================================================
# Total params: 12,233,232
# Trainable params: 12,233,232
# Non-trainable params: 0
# Total mult-adds (M): 127.70
# ===============================================================================================
# Input size (MB): 1.03
# Forward/backward pass size (MB): 16.08
# Params size (MB): 46.67
# Estimated Total Size (MB): 63.78
# ===============================================================================================


model = models.efficientnet_b4().cuda()
# summary(model, (3, 380, 380))
# ===============================================================================================
# Total params: 19,341,616
# Trainable params: 19,341,616
# Non-trainable params: 0
# Total mult-adds (M): 234.19
# ===============================================================================================
# Input size (MB): 1.65
# Forward/backward pass size (MB): 30.39
# Params size (MB): 73.78
# Estimated Total Size (MB): 105.82
# ===============================================================================================


model = models.efficientnet_b5().cuda()
# summary(model, (3, 456, 456))
# ===============================================================================================
# Total params: 30,389,784
# Trainable params: 30,389,784
# Non-trainable params: 0
# Total mult-adds (M): 418.15
# ===============================================================================================
# Input size (MB): 2.38
# Forward/backward pass size (MB): 45.11
# Params size (MB): 115.93
# Estimated Total Size (MB): 163.42
# ===============================================================================================


model = models.efficientnet_b6().cuda()
# summary(model, (3, 528, 528))
# ===============================================================================================
# Total params: 43,040,704
# Trainable params: 43,040,704
# Non-trainable params: 0
# Total mult-adds (M): 653.15
# ===============================================================================================
# Input size (MB): 3.19
# Forward/backward pass size (MB): 69.72
# Params size (MB): 164.19
# Estimated Total Size (MB): 237.10
# ===============================================================================================


model = models.efficientnet_b7().cuda()
# summary(model, (3, 600, 600))
# ===============================================================================================
# Total params: 66,347,960
# Trainable params: 66,347,960
# Non-trainable params: 0
# Total mult-adds (G): 1.00
# ===============================================================================================
# Input size (MB): 4.12
# Forward/backward pass size (MB): 102.00
# Params size (MB): 253.10
# Estimated Total Size (MB): 359.22
# ===============================================================================================



model = models.regnet_y_400mf().cuda()
# summary(model, (3, 224, 224))
# ====================================================================================================
# Total params: 4,344,144
# Trainable params: 4,344,144
# Non-trainable params: 0
# Total mult-adds (M): 26.81
# ====================================================================================================
# Input size (MB): 0.57
# Forward/backward pass size (MB): 6.13
# Params size (MB): 16.57
# Estimated Total Size (MB): 23.28
# ====================================================================================================


model = models.regnet_y_800mf().cuda()
# summary(model, (3, 224, 224))
# ====================================================================================================
# Total params: 6,432,512
# Trainable params: 6,432,512
# Non-trainable params: 0
# Total mult-adds (M): 34.13
# ====================================================================================================
# Input size (MB): 0.57
# Forward/backward pass size (MB): 6.13
# Params size (MB): 24.54
# Estimated Total Size (MB): 31.24
# ====================================================================================================


model = models.regnet_y_1_6gf().cuda()
# summary(model, (3, 224, 224))
# ====================================================================================================
# Total params: 11,202,430
# Trainable params: 11,202,430
# Non-trainable params: 0
# Total mult-adds (M): 52.83
# ====================================================================================================
# Input size (MB): 0.57
# Forward/backward pass size (MB): 6.13
# Params size (MB): 42.73
# Estimated Total Size (MB): 49.44
# ====================================================================================================


model = models.regnet_y_3_2gf().cuda()
# summary(model, (3, 224, 224))
# ====================================================================================================
# Total params: 19,436,338
# Trainable params: 19,436,338
# Non-trainable params: 0
# Total mult-adds (M): 83.86
# ====================================================================================================
# Input size (MB): 0.57
# Forward/backward pass size (MB): 6.13
# Params size (MB): 74.14
# Estimated Total Size (MB): 80.85
# ====================================================================================================


model = models.regnet_y_8gf().cuda()
# summary(model, (3, 224, 224))
# ====================================================================================================
# Total params: 39,381,472
# Trainable params: 39,381,472
# Non-trainable params: 0
# Total mult-adds (M): 162.07
# ====================================================================================================
# Input size (MB): 0.57
# Forward/backward pass size (MB): 6.13
# Params size (MB): 150.23
# Estimated Total Size (MB): 156.94
# ====================================================================================================


model = models.regnet_y_16gf().cuda()
# summary(model, (3, 224, 224))
# ====================================================================================================
# Total params: 83,590,140
# Trainable params: 83,590,140
# Non-trainable params: 0
# Total mult-adds (M): 335.78
# ====================================================================================================
# Input size (MB): 0.57
# Forward/backward pass size (MB): 6.13
# Params size (MB): 318.87
# Estimated Total Size (MB): 325.58
# ====================================================================================================


model = models.regnet_y_32gf().cuda()
# summary(model, (3, 224, 224))
# ====================================================================================================
# Total params: 145,046,770
# Trainable params: 145,046,770
# Non-trainable params: 0
# Total mult-adds (M): 579.45
# ====================================================================================================
# Input size (MB): 0.57
# Forward/backward pass size (MB): 6.13
# Params size (MB): 553.31
# Estimated Total Size (MB): 560.02
# ====================================================================================================


model = efficientnet_v2_s().cuda()
# summary(model, (3, 300, 300))
# ==========================================================================================
# Total params: 21,458,488
# Trainable params: 21,458,488
# Non-trainable params: 0
# Total mult-adds (G): 5.37
# ==========================================================================================
# Input size (MB): 1.03
# Forward/backward pass size (MB): 346.62
# Params size (MB): 81.86
# Estimated Total Size (MB): 429.51
# ==========================================================================================


model = efficientnet_v2_m().cuda()
# summary(model, (3, 384, 384))
# ==========================================================================================
# Total params: 54,139,356
# Trainable params: 54,139,356
# Non-trainable params: 0
# Total mult-adds (G): 15.90
# ==========================================================================================
# Input size (MB): 1.69
# Forward/backward pass size (MB): 877.18
# Params size (MB): 206.53
# Estimated Total Size (MB): 1085.39
# ==========================================================================================


model = efficientnet_v2_l().cuda()
# summary(model, (3, 384, 384))
# ==========================================================================================
# Total params: 118,515,272
# Trainable params: 118,515,272
# Non-trainable params: 0
# Total mult-adds (G): 36.25
# ==========================================================================================
# Input size (MB): 1.69
# Forward/backward pass size (MB): 1542.59
# Params size (MB): 452.10
# Estimated Total Size (MB): 1996.38
# ==========================================================================================