import torch
from torch import nn
import onnx
from onnxsim import simplify
import onnxruntime as ort
import numpy as np

from pytorch_self_cdist import my_cdist_p2_v2  # very fast


# [784, 384] [16385, 384]


class Cdist(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, y):
        return my_cdist_p2_v2(x, y)


x = torch.randn(1, 784, 384)
y = torch.randn(1, 16000, 384)
cdist = Cdist()
z = cdist(x, y)

onnx_path = 'cdist4.onnx'
torch.onnx.export(
    cdist,
    (x, y),
    onnx_path,
    opset_version=11,
    input_names=["x", "y"],
    output_names=["z"],
)
model_ = onnx.load(onnx_path)
# 简化模型
model_simp, check = simplify(model_)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp, onnx_path)
print("export onnx success!")


if True:
    ## onnxruntime
    net = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    # net = ort.InferenceSession(onnx_path1, providers=['CUDAExecutionProvider'], provider_options=[{'device_id': 0}])

    x = np.random.randn(1, 784, 384)
    x = x.astype(np.float32)
    y = np.random.randn(1, 16000, 384)
    y = y.astype(np.float32)


    input_names  = net.get_inputs()
    input_name1  = input_names[0].name
    input_name2  = input_names[1].name
    print(input_name1, input_name2) # x y
    output_names = net.get_outputs()
    output_name  = output_names[0].name
    print(output_name)              # z


    out = net.run(None,{input_name1: x, input_name2: y})
    print(out)
