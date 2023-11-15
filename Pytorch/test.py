import torch
import torchaudio
import torchvision


print(torch.__version__)
print(torchvision.__version__)
print(torchaudio.__version__)


# https://www.jianshu.com/p/996951e5e9f3
if torch.cuda.is_available() and torch.version.hip:
    print("do something specific for HIP")
elif torch.cuda.is_available() and torch.version.cuda:
    print("do something specific for CUDA")
