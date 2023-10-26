from re import X

import torch
import torchaudio
import torchvision

print(torch.__version__)
print(torchvision.__version__)
print(torchaudio.__version__)

x = torch.rand(2, 3, 4)

x1, x2 = x.unbind(0)
print(x1.shape, x2.shape)
# [3, 4] [3, 4]


x3, x4, x5 = x.unbind(1)
print(x3.shape, x4.shape, x4.shape)
# [2, 4] [2, 4] [2, 4]
