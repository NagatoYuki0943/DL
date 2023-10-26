import onnx
import cv2
import numpy as np

from utils import check_onnx


ONNX_PATH = "./resnet18.onnx"
# ONNX_PATH = "./twins_svt_small.onnx"  # transformer有问题
IMAGE_PATH = "../bus.jpg"
CUDA = False


# 检查onnx是否完好
check_onnx(ONNX_PATH)


# ---------------------------------------------#
#   dnn加载模型
# ---------------------------------------------#
net = cv2.dnn.readNetFromONNX(ONNX_PATH)
# net = cv2.dnn.readNet(onnx_path)
if CUDA:
    print("Attempty to use CUDA")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
else:
    print("Running on CPU")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


# ---------------------------------------------#
#   调整图片
# ---------------------------------------------#
image = cv2.imread(IMAGE_PATH)
inpWidth = 224
inpHeight = 224

mean = np.array((0.485, 0.456, 0.406)) * 255
std  = np.array((0.229, 0.224, 0.225)) * 255

# [H, W, C] -> [B, C, H, W] & BRG2RGB & 归一化等操作
# 当同时进行swapRB, scalefactor, mean, size操作时，优先按swapRB交换通道，然后按mean求减，其次按scalefactor比例相乘，最后按size进行resize操作
blob = cv2.dnn.blobFromImage(image,
                            swapRB=True,                    # 交换 Red 和 Blue 通道, BGR2RGB
                            mean=mean,                      # 用于各通道减去的均值
                            scalefactor=1.0 / 255,          # 图像各通道数值的缩放比例
                            # std=[0.229, 0.224, 0.225],    # 没有std，不要自己加
                            size=(inpWidth, inpHeight),     # resize图片大小 w,h
                            crop=False,                     # 图像裁剪,默认为False.当值为True时，先按比例缩放，然后从中心裁剪成size尺寸
                            ddepth=cv2.CV_32F               # 数据类型,可选 CV_32F 或者 CV_8U
                            )


# ---------------------------------------------#
#   推理
# ---------------------------------------------#
# 设置模型输入
net.setInput(blob)
# 返回2维numpy数组
out = net.forward()
print(out.shape)  # (1,10)


#---------------------------------------------#
#   numpy实现softmax
#---------------------------------------------#
def softmax(x, axis=0):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

out = softmax(out, axis=1)
# 最大值
print(out.max(axis=1))      # [0.12699042]
# 最大值下标
print(out.argmax(axis=1))   # [7