# script与trace区别 https://blog.csdn.net/weixin_41779359/article/details/109009192
# torchscript解读 https://zhuanlan.zhihu.com/p/486914187?utm_source=qq&utm_medium=social&utm_oi=1174092671630479360
# mmdeploy解读 https://mmdeploy.readthedocs.io/zh_CN/latest/tutorial/03_pytorch2onnx.html

"""
共有两种方法将pytorch模型转成torch script,一种是trace,另一种是script.
一般在模型内部没有控制流存在的话(if,for循环),直接用trace方法就可以了.
如果模型内部存在控制流,那就需要用到script方法了.

注意:
    1.如果内部判断是 `self.training` 判断是否是训练模式则可以使用trace,
      不过要将模型设置为 eval 才使用推理输出(对于trace和script都相同),
      即使使用 inference_mode 也依然是训练的输出, 所以必须使用 eval
    2.两种导出模式都分为cpu和cuda
      载入cuda模型不需要将模型转换到cuda上
"""


import torch
from torch import Tensor


x = torch.ones(1, 3, 224, 224)


#-------------------------------------------#
#   trace方法
#-------------------------------------------#
class Trace(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, 4, stride=4)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)

model = Trace()
model.eval()                    # important, don't forget!!!

with torch.inference_mode():
    y = model(x)
print(y.size())                 # [1, 64, 56, 56]

#-------------------------------------------#
# cpu和gpu分开导出 refer: https://pytorch.org/docs/stable/jit.html#frequently-asked-questions
# cpu
trace_model = torch.jit.trace(model, example_inputs=x)
# print(trace_module.code)          # 查看模型结构
torch.jit.save(trace_model, "m_cpu.torchtrace")

# gpu
if torch.cuda.is_available():
    trace_model = torch.jit.trace(model.cuda(), example_inputs=x.cuda())
    torch.jit.save(trace_model, "m_gpu.torchtrace")

#-------------------------------------------#
# cpu和gpu分开导入, gpu模型不需要转换为cuda
# cpu
trace_model_ = torch.jit.load("m_cpu.torchtrace")
with torch.inference_mode():
    y_ = trace_model_(x)
print(y_.size())                # [1, 64, 56, 56]
print(torch.all(y==y_))         # True

# gpu
if torch.cuda.is_available():
    trace_model_ = torch.jit.load("m_gpu.torchtrace")
    with torch.inference_mode():
        y_ = trace_model_(x.cuda())
    print(y_.size())                # [1, 64, 56, 56]
    print(torch.all(y==y_.cpu()))   # False
    print(y[0, 0, 0, 0] - y_.cpu()[0, 0, 0, 0])   # -1.1921e-07 说明cpu和gpu结果略有不同


#-------------------------------------------#
#   测试train和eval的差异
#   如果内部判断是 `self.training` 判断是否是训练模式则可以使用trace,不过要将模型设置为 eval
#   即使使用 inference_mode 也依然是训练的输出, 所以必须使用 eval
#-------------------------------------------#
class Train_Eval(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, 4, stride=4)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            return self.conv(x)
        else:
            return self.conv(x).flatten(2).transpose(1, 2)

train_eval = Train_Eval()

#-------------------------------------------#
# 不设置eval会按照训练模式导出
with torch.inference_mode():    # 使用 inference_mode 也依然是训练的输出
    trace_model = torch.jit.trace(train_eval, example_inputs=x)
    y = trace_model(x)
print(y.size())                 # [1, 64, 56, 56]

#-------------------------------------------#
# 设置eval会使用推理模式导出
train_eval.eval()               # important, don't forget!!!
trace_model = torch.jit.trace(train_eval, example_inputs=x)
with torch.inference_mode():
    y = trace_model(x)
print(y.size())                 # [1, 3136, 64]
