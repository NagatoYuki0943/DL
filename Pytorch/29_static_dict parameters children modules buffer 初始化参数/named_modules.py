from torchvision import models


model = models.resnet18()

# children() 只会遍历模型的子层                                 放入Sequential中可以执行
# modules()  迭代遍历模型的所有子层，所有子层即指nn.Module子类    放入Sequential中不能执行,因为递归返回的太多
for name, module in model.named_modules():
    print(name)
print()
# conv1
# bn1
# relu
# maxpool
# layer1
# layer1.0
# layer1.0.conv1
# layer1.0.bn1
# layer1.0.relu
# layer1.0.conv2
# layer1.0.bn2
# layer1.1
# layer1.1.conv1
# layer1.1.bn1
# layer1.1.relu
# layer1.1.conv2
# layer1.1.bn2
# layer2
# layer2.0
# layer2.0.conv1
# layer2.0.bn1
# layer2.0.relu
# layer2.0.conv2
# layer2.0.bn2
# layer2.0.downsample
# layer2.0.downsample.0
# layer2.0.downsample.1
# layer2.1
# layer2.1.conv1
# layer2.1.bn1
# layer2.1.relu
# layer2.1.conv2
# layer2.1.bn2
# layer3
# layer3.0
# layer3.0.conv1
# layer3.0.bn1
# layer3.0.relu
# layer3.0.conv2
# layer3.0.bn2
# layer3.0.downsample
# layer3.0.downsample.0
# layer3.0.downsample.1
# layer3.1
# layer3.1.conv1
# layer3.1.bn1
# layer3.1.relu
# layer3.1.conv2
# layer3.1.bn2
# layer4
# layer4.0
# layer4.0.conv1
# layer4.0.bn1
# layer4.0.relu
# layer4.0.conv2
# layer4.0.bn2
# layer4.0.downsample
# layer4.0.downsample.0
# layer4.0.downsample.1
# layer4.1
# layer4.1.conv1
# layer4.1.bn1
# layer4.1.relu
# layer4.1.conv2
# layer4.1.bn2
# avgpool
# fc


for module in model.modules():
    print(module)
print()
# ResNet(
#   (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#   (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (relu): ReLU(inplace=True)
#   (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
#   (layer1): Sequential(
#     (0): BasicBlock(
#       (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (1): BasicBlock(
#       (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (layer2): Sequential(
#     (0): BasicBlock(
#       (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (downsample): Sequential(
#         (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
#         (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): BasicBlock(
#       (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (layer3): Sequential(
#     (0): BasicBlock(
#       (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (downsample): Sequential(
#         (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
#         (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): BasicBlock(
#       (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (layer4): Sequential(
#     (0): BasicBlock(
#       (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (downsample): Sequential(
#         (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
#         (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): BasicBlock(
#       (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
#   (fc): Linear(in_features=512, out_features=1000, bias=True)
# )
# Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# ReLU(inplace=True)
# MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
# Sequential(
#   (0): BasicBlock(
#     (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (relu): ReLU(inplace=True)
#     (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   )
#   (1): BasicBlock(
#     (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (relu): ReLU(inplace=True)
#     (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   )
# )
# BasicBlock(
#   (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#   (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (relu): ReLU(inplace=True)
#   (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#   (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# )
# Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
# BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# ReLU(inplace=True)
# Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
# BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# BasicBlock(
#   (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#   (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (relu): ReLU(inplace=True)
#   (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#   (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# )
# Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
# BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# ReLU(inplace=True)
# Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
# BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# Sequential(
#   (0): BasicBlock(
#     (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#     (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (relu): ReLU(inplace=True)
#     (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (downsample): Sequential(
#       (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
#       (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (1): BasicBlock(
#     (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (relu): ReLU(inplace=True)
#     (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   )
# )
# BasicBlock(
#   (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#   (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (relu): ReLU(inplace=True)
#   (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#   (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (downsample): Sequential(
#     (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
#     (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   )
# )
# Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
# BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# ReLU(inplace=True)
# Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
# BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# Sequential(
#   (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
#   (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# )
# Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
# BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# BasicBlock(
#   (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#   (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (relu): ReLU(inplace=True)
#   (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#   (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# )
# Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
# BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# ReLU(inplace=True)
# Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
# BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# Sequential(
#   (0): BasicBlock(
#     (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#     (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (relu): ReLU(inplace=True)
#     (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (downsample): Sequential(
#       (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
#       (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (1): BasicBlock(
#     (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (relu): ReLU(inplace=True)
#     (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   )
# )
# BasicBlock(
#   (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#   (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (relu): ReLU(inplace=True)
#   (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#   (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (downsample): Sequential(
#     (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
#     (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   )
# )
# Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
# BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# ReLU(inplace=True)
# Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
# BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# Sequential(
#   (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
#   (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# )
# Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
# BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# BasicBlock(
#   (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#   (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (relu): ReLU(inplace=True)
#   (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#   (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# )
# Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
# BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# ReLU(inplace=True)
# Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
# BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# Sequential(
#   (0): BasicBlock(
#     (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#     (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (relu): ReLU(inplace=True)
#     (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (downsample): Sequential(
#       (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
#       (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (1): BasicBlock(
#     (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (relu): ReLU(inplace=True)
#     (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   )
# )
# BasicBlock(
#   (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#   (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (relu): ReLU(inplace=True)
#   (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#   (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (downsample): Sequential(
#     (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
#     (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   )
# )
# Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
# BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# ReLU(inplace=True)
# Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
# BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# Sequential(
#   (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
#   (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# )
# Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
# BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# BasicBlock(
#   (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#   (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (relu): ReLU(inplace=True)
#   (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#   (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# )
# Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
# BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# ReLU(inplace=True)
# Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
# BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# AdaptiveAvgPool2d(output_size=(1, 1))
# Linear(in_features=512, out_features=1000, bias=True)
