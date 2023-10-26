# 提取模型需要的部分,忽略不要的

> 需要pytorch==1.10以上版本和配合的torchvision

```python
import torchvision
#   feature_extraction
from torchvision.models.feature_extraction import create_feature_extractor

# vgg16
backbone = torchvision.models.vgg16_bn(pretrained=False)
# print(backbone)
#----------------------------------------------------#
#   提取需要的节点和之前的节点,删除后面的部分
#----------------------------------------------------#
backbone = create_feature_extractor(backbone, return_nodes={"features.42": "0"}) # 提取的节点: 新名称
# out = backbone(torch.rand(1, 3, 224, 224))
# print(out["0"].shape)
```

