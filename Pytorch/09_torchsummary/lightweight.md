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





# lightweight模型大小对比

| model              | Total params | Total mult-adds | Params size (MB) |
| ------------------ | ------------ | --------------- | ---------------- |
| squeezenet1_1      | 1,235,496    | 351.10M         | 4.71             |
| shufflenet_v2_x2.0 | 7,393,996    | 595.25M         | 28.21            |
| mobilenet_v2       | 3,504,872    | 158.62M         | 13.37            |
| mobilenet_v3_small | 2,542,856    | 14.92M          | 9.70             |
| mobilenet_v3_large | 5,483,032    | 29.47M          | 20.92            |
|                    |              |                 |                  |
|                    |              |                 |                  |
|                    |              |                 |                  |
|                    |              |                 |                  |



# lightweight

## squeezenet1_1

```python
model = models.squeezenet1_1()
summary(model, (3, 224, 224))
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
```

## shufflenet_v2_x2.0

```python
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
```

## mobilenet_v2

```python
from torchvision.models import mobilenet
model = mobilenet.mobilenet_v2()
# summary(model, (3, 224, 224))
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
```

## mobilenet_v3_small

```python
from torchvision.models import mobilenet
model = mobilenet.mobilenet_v3_small()
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
```

## mobilenet_v3_large

```python
from torchvision.models import mobilenet
model = mobilenet.mobilenet_v3_large()
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
```



