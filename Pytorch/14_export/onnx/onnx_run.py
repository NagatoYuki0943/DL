# https://github.com/microsoft/onnxruntime/blob/main/docs/python/inference/tutorial.rst
from PIL import Image
from torchvision import transforms
import onnxruntime as ort
import numpy as np

from utils import check_onnx


print(ort.__version__)
# print("onnxruntime all providers:", ort.get_all_providers())
print("onnxruntime available providers:", ort.get_available_providers())
# ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
print(ort.get_device())
# GPU


ONNX_PATH = "./resnet18.onnx"
# ONNX_PATH = "./twins_svt_small.onnx"  # transformer有问题
IMAGE_PATH = "../bus.jpg"


# 检查onnx是否完好
check_onnx(ONNX_PATH)


def get_onnx_model(onnx_path: str, mode: str="cpu") -> ort.InferenceSession:
    """获取onnxruntime模型
    Args:
        onnx_path (str): 模型路径
        mode (str, optional): cpu cuda tensorrt. Defaults to cpu.
    Returns:
        ort.InferenceSession: 模型session
    """
    mode = mode.lower()
    assert mode in ["cpu", "cuda", "tensorrt"], "onnxruntime only support cpu, cuda and tensorrt inference."
    print(f"inference with {mode} !")

    so = ort.SessionOptions()
    so.log_severity_level = 3
    providers = {
        "cpu":  ['CPUExecutionProvider'],
        # cuda
        # https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
        "cuda": [
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,            # 2GB
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }),
                'CPUExecutionProvider',
            ],
        # tensorrt
        # https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html
        # it is recommended you also register CUDAExecutionProvider to allow Onnx Runtime to assign nodes to CUDA execution provider that TensorRT does not support.
        # set providers to ['TensorrtExecutionProvider', 'CUDAExecutionProvider'] with TensorrtExecutionProvider having the higher priority.
        "tensorrt": [
                ('TensorrtExecutionProvider', {
                    'device_id': 0,
                    'trt_max_workspace_size': 2 * 1024 * 1024 * 1024,   # 2GB
                    'trt_fp16_enable': False,
                }),
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,            # 2GB
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                })
            ]
    }[mode]

    model = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)

    #---------------------------------------------#
    #   查看model中的内容
    #   get_inputs()返回对象，[0]返回名字
    #---------------------------------------------#
    print("model outputs: ", model.get_inputs())    # 列表 [<onnxruntime.capi.onnxruntime_pybind11_state.NodeArg object at 0x0000024F2136BC70>]
    print(model.get_inputs()[0])                    # NodeArg(name='input', type='tensor(float)', shape=['batch_size', 'channel', 'height', 'width'])
    print(model.get_inputs()[0].name)               # image
    print(model.get_inputs()[0].type)               # tensor(float)
    print(model.get_inputs()[0].shape, "\n")        # ['batch_size', '3', 'height', 'width']

    print("model outputs: ", model.get_outputs())   # 列表 [<onnxruntime.capi.onnxruntime_pybind11_state.NodeArg object at 0x000002475CE8EA30>]
    print(model.get_outputs()[0])                   # NodeArg(name='output', type='tensor(float)', shape=['batch_size', 50])
    print(model.get_outputs()[0].name)              # classes
    print(model.get_outputs()[0].type)              # tensor(float)
    print(model.get_outputs()[0].shape, "\n")       # ['batch_size', 10]

    return model


def get_transform(resize=224):
    return transforms.Compose([
            transforms.Resize(int(resize*1.25)),
            transforms.RandomCrop((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])


#---------------------------------------------#
#   读取图像并转化为tensor
#---------------------------------------------#
image = Image.open(IMAGE_PATH)
resize = 224
transform = get_transform(resize)
input = transform(img=image)

# 转化为tensor并扩展维度 [C, H, W] -> [B, C, H, W]
input = np.array(input.unsqueeze(dim=0))
print(input.shape)  # 1, 3, 224, 224

#---------------------------------------------#
#   numpy实现softmax
#---------------------------------------------#
def softmax(x, axis=0):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


#---------------------------------------------#
#   推理
#   run参数：
#       参数1：返回值列表，[output_name] 来规定想返回的值,可以为None,返回全部outputs
#       参数2： 输入值的字典 {input_name: input}
#---------------------------------------------#
model       = get_onnx_model(ONNX_PATH, "cuda")
inputs      = model.get_inputs()
input_name  = inputs[0].name
outputs     = model.get_outputs()
output_name = outputs[0].name
# 参数1是返回数据的名字
# 返回一个列表,每一个数据是一个3维numpy数组
# outs = model.run([output_name], {input_name: input})
outs = model.run(None, {input_name: input})
print(type(outs))   # <class 'list'>
out = outs[0]

print(out.shape)   # (1, 10)

out = softmax(out, axis=1)
# 最大值
print(out.max(axis=1))      # [0.544652]
# 最大值下标
print(out.argmax(axis=1))   # [4]
