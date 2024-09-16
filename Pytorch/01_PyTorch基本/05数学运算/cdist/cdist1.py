import torch
from torch import nn
import onnx
from onnxsim import simplify
import onnxruntime as ort
import numpy as np


"""使用register_buffer或者保存局部tensor对于onnx是完全相同的
"""


class Cdist(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("y", torch.ones(16000, 384))

    def forward(self, x):
        return torch.cdist(x, self.y, p=2)


x = torch.randn(784, 384)
cdist = Cdist()
z = cdist(x)

onnx_path = "cdist1.onnx"
torch.onnx.export(
    cdist,
    x,
    onnx_path,
    opset_version=11,
    input_names=["x"],
    output_names=["z"],
)
model_ = onnx.load(onnx_path)
# 简化模型
model_simp, check = simplify(model_)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp, onnx_path)
print("export onnx success!")


if False:
    ## onnxruntime
    net = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    # net = ort.InferenceSession(onnx_path1, providers=['CUDAExecutionProvider'], provider_options=[{'device_id': 0}])

    x = np.random.randn(784, 384)
    x = x.astype(np.float32)

    input_name = net.get_inputs()[0].name
    output_name = net.get_outputs()[0].name

    out = net.run(None, {input_name: x})
    print(out)
