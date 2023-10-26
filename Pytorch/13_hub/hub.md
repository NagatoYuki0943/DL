# load_state_dict_from_url

```python
from torch.hub import load_state_dict_from_url
from torchvision.models import resnet18

state_dict = torch.hub.load_state_dict_from_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth', model_dir="./model_data")
model = resnet18()
model.load_state_dict(state_dict)
```

# torch.hub.load() 加载预训练模型

https://pytorch.org/hub/research-models

https://github.com/pytorch/hub

```python
import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, custom

# Images
img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
```
