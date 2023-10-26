from torchvision import models


model = models.resnet18()


for name, parameter in model.named_parameters():
    print(name)
print()
# conv1.weight
# bn1.weight
# bn1.bias
# layer1.0.conv1.weight
# layer1.0.bn1.weight
# layer1.0.bn1.bias
# layer1.0.conv2.weight
# layer1.0.bn2.weight
# layer1.0.bn2.bias
# layer1.1.conv1.weight
# layer1.1.bn1.weight
# layer1.1.bn1.bias
# layer1.1.conv2.weight
# layer1.1.bn2.weight
# layer1.1.bn2.bias
# layer2.0.conv1.weight
# layer2.0.bn1.weight
# layer2.0.bn1.bias
# layer2.0.conv2.weight
# layer2.0.bn2.weight
# layer2.0.bn2.bias
# layer2.0.downsample.0.weight
# layer2.0.downsample.1.weight
# layer2.0.downsample.1.bias
# layer2.1.conv1.weight
# layer2.1.bn1.weight
# layer2.1.bn1.bias
# layer2.1.conv2.weight
# layer2.1.bn2.weight
# layer2.1.bn2.bias
# layer3.0.conv1.weight
# layer3.0.bn1.weight
# layer3.0.bn1.bias
# layer3.0.conv2.weight
# layer3.0.bn2.weight
# layer3.0.bn2.bias
# layer3.0.downsample.0.weight
# layer3.0.downsample.1.weight
# layer3.0.downsample.1.bias
# layer3.1.conv1.weight
# layer3.1.bn1.weight
# layer3.1.bn1.bias
# layer3.1.conv2.weight
# layer3.1.bn2.weight
# layer3.1.bn2.bias
# layer4.0.conv1.weight
# layer4.0.bn1.weight
# layer4.0.bn1.bias
# layer4.0.conv2.weight
# layer4.0.bn2.weight
# layer4.0.bn2.bias
# layer4.0.downsample.0.weight
# layer4.0.downsample.1.weight
# layer4.0.downsample.1.bias
# layer4.1.conv1.weight
# layer4.1.bn1.weight
# layer4.1.bn1.bias
# layer4.1.conv2.weight
# layer4.1.bn2.weight
# layer4.1.bn2.bias
# fc.weight
# fc.bias


for parameter in model.parameters():
    pass
    # print(parameter)


# 冻结部分权重
for name, parameter in model.named_parameters():
    if "fc" not in name:
        parameter.requires_grad = False
    else:
        print(name)
        # fc.weight
        # fc.bias

parameters = [param for param in model.parameters() if param.requires_grad]

from torch.optim import AdamW
AdamW(parameters, 0.001)
