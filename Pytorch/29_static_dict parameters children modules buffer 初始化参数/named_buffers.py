from torchvision import models


model = models.resnet18()


for name, buffer in model.named_buffers():
    print(name)
print()
# bn1.running_mean
# bn1.running_var
# bn1.num_batches_tracked
# layer1.0.bn1.running_mean
# layer1.0.bn1.running_var
# layer1.0.bn1.num_batches_tracked
# layer1.0.bn2.running_mean
# layer1.0.bn2.running_var
# layer1.0.bn2.num_batches_tracked
# layer1.1.bn1.running_mean
# layer1.1.bn1.running_var
# layer1.1.bn1.num_batches_tracked
# layer1.1.bn2.running_mean
# layer1.1.bn2.running_var
# layer1.1.bn2.num_batches_tracked
# layer2.0.bn1.running_mean
# layer2.0.bn1.running_var
# layer2.0.bn1.num_batches_tracked
# layer2.0.bn2.running_mean
# layer2.0.bn2.running_var
# layer2.0.bn2.num_batches_tracked
# layer2.0.downsample.1.running_mean
# layer2.0.downsample.1.running_var
# layer2.0.downsample.1.num_batches_tracked
# layer2.1.bn1.running_mean
# layer2.1.bn1.running_var
# layer2.1.bn1.num_batches_tracked
# layer2.1.bn2.running_mean
# layer2.1.bn2.running_var
# layer2.1.bn2.num_batches_tracked
# layer3.0.bn1.running_mean
# layer3.0.bn1.running_var
# layer3.0.bn1.num_batches_tracked
# layer3.0.bn2.running_mean
# layer3.0.bn2.running_var
# layer3.0.bn2.num_batches_tracked
# layer3.0.downsample.1.running_mean
# layer3.0.downsample.1.running_var
# layer3.0.downsample.1.num_batches_tracked
# layer3.1.bn1.running_mean
# layer3.1.bn1.running_var
# layer3.1.bn1.num_batches_tracked
# layer3.1.bn2.running_mean
# layer3.1.bn2.running_var
# layer3.1.bn2.num_batches_tracked
# layer4.0.bn1.running_mean
# layer4.0.bn1.running_var
# layer4.0.bn1.num_batches_tracked
# layer4.0.bn2.running_mean
# layer4.0.bn2.running_var
# layer4.0.bn2.num_batches_tracked
# layer4.0.downsample.1.running_mean
# layer4.0.downsample.1.running_var
# layer4.0.downsample.1.num_batches_tracked
# layer4.1.bn1.running_mean
# layer4.1.bn1.running_var
# layer4.1.bn1.num_batches_tracked
# layer4.1.bn2.running_mean
# layer4.1.bn2.running_var
# layer4.1.bn2.num_batches_tracked


for buffer in model.buffers():
    pass
    # print(buffer)
print()
