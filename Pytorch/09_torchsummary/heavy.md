# torchsummary

torchsummary其主要是用来计算网络的计算参数等信息的



## 安装

`pip install torchsummary`



## 使用

```python
import  torch
from torch import nn
from torchvision.models import resnet18
from torchsummary import summary

model = resnet18()
model.fc = nn.Linear(model.fc.in_features, 5)

#              输出图片大小
summary(model, (3, 224, 224))
```

## 输出

```python

# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param # 
# ================================================================
#             Conv2d-1         [-1, 64, 112, 112]           9,408 
#        BatchNorm2d-2         [-1, 64, 112, 112]             128 
#               ReLU-3         [-1, 64, 112, 112]               0 
#          MaxPool2d-4           [-1, 64, 56, 56]               0 
#             Conv2d-5           [-1, 64, 56, 56]          36,864 
#        BatchNorm2d-6           [-1, 64, 56, 56]             128 
#               ReLU-7           [-1, 64, 56, 56]               0 
#             Conv2d-8           [-1, 64, 56, 56]          36,864 
#        BatchNorm2d-9           [-1, 64, 56, 56]             128 
#              ReLU-10           [-1, 64, 56, 56]               0 
#        BasicBlock-11           [-1, 64, 56, 56]               0 
#            Conv2d-12           [-1, 64, 56, 56]          36,864 
#       BatchNorm2d-13           [-1, 64, 56, 56]             128 
#              ReLU-14           [-1, 64, 56, 56]               0 
#            Conv2d-15           [-1, 64, 56, 56]          36,864
#       BatchNorm2d-16           [-1, 64, 56, 56]             128
#              ReLU-17           [-1, 64, 56, 56]               0
#        BasicBlock-18           [-1, 64, 56, 56]               0
#            Conv2d-19          [-1, 128, 28, 28]          73,728
#       BatchNorm2d-20          [-1, 128, 28, 28]             256
#              ReLU-21          [-1, 128, 28, 28]               0
#            Conv2d-22          [-1, 128, 28, 28]         147,456
#       BatchNorm2d-23          [-1, 128, 28, 28]             256
#            Conv2d-24          [-1, 128, 28, 28]           8,192
#       BatchNorm2d-25          [-1, 128, 28, 28]             256
#              ReLU-26          [-1, 128, 28, 28]               0
#        BasicBlock-27          [-1, 128, 28, 28]               0
#            Conv2d-28          [-1, 128, 28, 28]         147,456
#       BatchNorm2d-29          [-1, 128, 28, 28]             256
#              ReLU-30          [-1, 128, 28, 28]               0
#            Conv2d-31          [-1, 128, 28, 28]         147,456
#       BatchNorm2d-32          [-1, 128, 28, 28]             256
#              ReLU-33          [-1, 128, 28, 28]               0
#        BasicBlock-34          [-1, 128, 28, 28]               0
#            Conv2d-35          [-1, 256, 14, 14]         294,912
#       BatchNorm2d-36          [-1, 256, 14, 14]             512
#              ReLU-37          [-1, 256, 14, 14]               0
#            Conv2d-38          [-1, 256, 14, 14]         589,824
#       BatchNorm2d-39          [-1, 256, 14, 14]             512
#            Conv2d-40          [-1, 256, 14, 14]          32,768
#       BatchNorm2d-41          [-1, 256, 14, 14]             512
#              ReLU-42          [-1, 256, 14, 14]               0
#        BasicBlock-43          [-1, 256, 14, 14]               0
#            Conv2d-44          [-1, 256, 14, 14]         589,824
#       BatchNorm2d-45          [-1, 256, 14, 14]             512
#              ReLU-46          [-1, 256, 14, 14]               0
#            Conv2d-47          [-1, 256, 14, 14]         589,824
#       BatchNorm2d-48          [-1, 256, 14, 14]             512
#              ReLU-49          [-1, 256, 14, 14]               0
#        BasicBlock-50          [-1, 256, 14, 14]               0
#            Conv2d-51            [-1, 512, 7, 7]       1,179,648
#       BatchNorm2d-52            [-1, 512, 7, 7]           1,024
#              ReLU-53            [-1, 512, 7, 7]               0
#            Conv2d-54            [-1, 512, 7, 7]       2,359,296
#       BatchNorm2d-55            [-1, 512, 7, 7]           1,024
#            Conv2d-56            [-1, 512, 7, 7]         131,072
#       BatchNorm2d-57            [-1, 512, 7, 7]           1,024
#              ReLU-58            [-1, 512, 7, 7]               0
#        BasicBlock-59            [-1, 512, 7, 7]               0
#            Conv2d-60            [-1, 512, 7, 7]       2,359,296
#       BatchNorm2d-61            [-1, 512, 7, 7]           1,024
#              ReLU-62            [-1, 512, 7, 7]               0
#            Conv2d-63            [-1, 512, 7, 7]       2,359,296
#       BatchNorm2d-64            [-1, 512, 7, 7]           1,024
#              ReLU-65            [-1, 512, 7, 7]               0
#        BasicBlock-66            [-1, 512, 7, 7]               0
# AdaptiveAvgPool2d-67            [-1, 512, 1, 1]               0
#            Linear-68                    [-1, 5]           2,565
# ================================================================
# Total params: 11,179,077
# Trainable params: 11,179,077
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 62.79
# Params size (MB): 42.64
# Estimated Total Size (MB): 106.00
# ----------------------------------------------------------------
```

# Total params 与 Params size (MB) 计算方法

$$
Total \; params * 32 / 8 / 1024 / 1024 = Total \; params / 262144  = Params \; size (MB)
$$



# 模型大小对比(num_classes=Default)

| model               | Total params | Total mult-adds | Params size (MB) |
| ------------------- | ------------ | --------------- | ---------------- |
| alexnet             | 61,100,840   | 775.28M         | 233.08           |
| vgg11               | 132,863,336  | 7.74G           | 506.83           |
| vgg11_bn            | 132,868,840  | 7.74G           | 506.85           |
| vgg13               | 133,047,848  | 11.44G          | 507.54           |
| vgg13_bn            | 133,053,736  | 11.44G          | 507.56           |
| vgg16               | 138,357,544  | 15.61G          | 527.79           |
| vgg16_bn            | 138,365,992  | 15.61G          | 527.82           |
| vgg19               | 143,667,240  | 19.78G          | 548.05           |
| vgg19_bn            | 143,678,248  | 19.78G          | 548.09           |
| googlenet           | 6,624,904    | 2.60G           | 25.27            |
| inception_v3        | 23,834,568   | 5.76G           | 90.92            |
| inception_v4        | 42,679,816   | 1.85G           | 162.81           |
| inception_resnet_v2 | 55,843,464   | 9.16G           | 213.03           |
| xception            | 22,855,952   | 4.60G           | 87.19            |
| resnet18            | 11,689,512   | 1.84G           | 44.59            |
| resnet34            | 21,797,672   | 3.71G           | 83.15            |
| resnet50            | 25,557,032   | 4.14G           | 97.49            |
| resnet101           | 44,549,160   | 7.89G           | 169.94           |
| resnet152           | 60,192,808   | 11.63G          | 229.62           |
| resnext50_32x4d     | 25,028,904   | 4.28G           | 95.48            |
| resnext101_32x8d    | 88,791,336   | 16.59G          | 338.71           |
| densenet121         | 7,978,856    | 2.85GG          | 30.44            |
| densenet169         | 14,149,480   | 3.40G           | 53.98            |
| densenet201         | 20,013,928   | 4.34G           | 76.35            |
| densenet161         | 28,681,000   | 7.80G           | 109.41           |
| efficientnet_v1_b0  | 5,288,548    | 48.55M          | 20.17            |
| efficientnet_v1_b1  | 7,794,184    | 66.27M          | 29.73            |
| efficientnet_v1_b2  | 9,109,994    | 87.18M          | 34.75            |
| efficientnet_v1_b3  | 12,233,232   | 127.70M         | 46.67            |
| efficientnet_v1_b4  | 19,341,616   | 234.19M         | 73.78            |
| efficientnet_v1_b5  | 30,389,784   | 418.15M         | 115.93           |
| efficientnet_v1_b6  | 43,040,704   | 653.15M         | 164.19           |
| efficientnet_v1_b7  | 66,347,960   | 1.00G           | 253.10           |
| regnet_y_400mf      | 4,344,144    | 26.81M          | 16.57            |
| regnet_y_800mf      | 6,432,512    | 34.13M          | 24.54            |
| regnet_y_1_6gf      | 11,202,430   | 52.83M          | 42.73            |
| regnet_y_3_2gf      | 19,436,338   | 83.86M          | 74.14            |
| regnet_y_8gf        | 39,381,472   | 162.07M         | 150.23           |
| regnet_y_16gf       | 83,590,140   | 335.78M         | 318.87           |
| regnet_y_32gf       | 145,046,770  | 579.45M         | 553.31           |
| efficientnet_v2_s   | 21,458,488   | 5.37G           | 81.86            |
| efficientnet_v2_m   | 54,139,356   | 15.90G          | 206.53           |
| efficientnet_v2_l   | 118,515,272  | 36.25G          | 452.10           |
| convnext_tiny       | 28,589,128   | 297.34M         | 109.06           |
| convnext_small      | 50,223,688   | 383.66M         | 191.59           |
| convnext_base       | 88,591,464   | 673.75M         | 337.95           |
| convnext_large      | 197,767,336  | 1.50G           | 754.42           |
|                     |              |                 |                  |
---



# alexnet

## alexnet

```python
model = models.alexnet()
model = model.cuda()
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

```



# vgg

##  vgg11

```python
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
```



## vgg11_bn

```python
model = models.vgg11_bn()
model = model.cuda()
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
```

## vgg13

```python
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
```

## vgg13_bn

```python
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
```

## vgg16

```python
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
```

## vgg16_bn

```python
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
```

## vgg19

```python
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
```

## vgg19_bn

```python
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
```

# inception

## googlenet

```python
model = models.googlenet()
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
```

## inception_v3

```python
model = models.inception_v3(init_weights = True).cuda()
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
```

## inception_v4

```python
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
```

## inceptionresnetv2

```python
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
```

## xception

```python
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
```

# resnet

## resnet18

```python
model = models.resnet18().cuda()
#              输入图片大小
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
```

## resnet34

```python
model = models.resnet34()
model = model.cuda()
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
```

## resnet50

```python
model = models.resnet50()
model = model.cuda()
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
```

## resnet101

```python
model = models.resnet101()
model = model.cuda()
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
```

## resnet152

```python
model = models.resnet152().cuda()
summary(model, (3, 224, 224))
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
```

## resnext50_32x4d

```python
model = models.resnext50_32x4d().cuda()
summary(model, (3, 224, 224))
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
```

## resnext101_32x8d

```python
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
```

# densenet

## densenet121

```python
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
```

## densenet169

```python
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
```

## densenet201

```python
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
```

## densenet161

```python
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
```

# efficientnet v1

## efficientnet_b0

```python
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
```

## efficientnet_b1

```python
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
```

## efficientnet_b2

```python
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
```

## efficientnet_b3

```python
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
```

## efficientnet_b4

```python
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
```

## efficientnet_b5

```python
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
```

## efficientnet_b6

```python
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
```

## efficientnet_b7

```python
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
```

# regnet

## regnet_y_400mf

```python
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
```

## regnet_y_800mf

```python
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
```

##  regnet_y_1_6gf

```python
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
```

## regnet_y_3_2gf

```python
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
```

## regnet_y_8gf

```python
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
```

## regnet_y_16gf

```python
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
```

## regnet_y_32gf

```python
model = models.regnet_y_32gf().cuda()
summary(model, (3, 224, 224))
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
```

# efficientnet v2

## efficientnet_v2_s

```python
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
```

## efficientnet_v2_m

```python
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
```

## efficientnet_v2_l

```python
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
```





# convnext

## convnext_tiny

```python
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
```

## convnext_small

```python
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
```

## convnext_base

```python
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
```

## convnext_large

```python
model = models.convnext_large().cuda()
summary(model, (3, 224, 224))
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
```
