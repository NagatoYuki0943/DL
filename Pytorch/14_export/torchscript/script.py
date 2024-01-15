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
    2.两种导出模式都分为cpu和cuda,载入cuda模型不需要将模型转换到cuda上,
      支持 `to(device) cpu() cuda()` 方法转移到指定设备
"""


import torch
from torch import Tensor


x1 = torch.ones(1, 1, 224, 224)
x3 = torch.ones(1, 3, 224, 224)


#-------------------------------------------#
#   script方法
#-------------------------------------------#
class Script(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, 4, stride=4)
        self.conv2 = torch.nn.Conv2d(3, 64, 4, stride=4)

    def forward(self, x):
        if x.size(1) == 1:
            return self.conv1(x)
        else:
            return self.conv2(x)


script_model = Script()
script_model.eval()                 # important, don't forget!!!

with torch.inference_mode():
    y1 = script_model(x1)
    y3 = script_model(x3)
print(y1.size(), y3.size())         # [1, 64, 56, 56]   [1, 64, 56, 56]


#------------------------------------#
# trace导出有控制流存在的模型会报错
try:
    with torch.inference_mode():
        trace_model = torch.jit.trace(script_model, example_inputs=x3)
except:
    print("Trace export error!")    # Trace export error!


#-------------------------------------------#
# cpu和gpu分开导出 refer: https://pytorch.org/docs/stable/jit.html#frequently-asked-questions
# cpu
script_model = torch.jit.script(script_model)
# print(script_module.code)         # 查看模型结构
torch.jit.save(script_model, "m_cpu.torchscript")


# gpu
if torch.cuda.is_available():
    script_model = torch.jit.script(script_model.cuda())
    # print(script_module.code)         # 查看模型结构
    torch.jit.save(script_model, "m_gpu.torchscript")


#-------------------------------------------#
# cpu和gpu分开导入, gpu模型不需要转换为cuda
# cpu
script_model_ = torch.jit.load("m_cpu.torchscript")
with torch.inference_mode():
    y1_ = script_model_(x1)
    y3_ = script_model_(x3)
print(y1_.size(), y3_.size())       # [1, 64, 56, 56]   [1, 64, 56, 56]
print(torch.all(y1==y1_))           # True
print(torch.all(y3==y3_))           # True


# gpu
if torch.cuda.is_available():
    script_model_ = torch.jit.load("m_gpu.torchscript")
    with torch.inference_mode():
        y1_ = script_model_(x1.cuda())
        y3_ = script_model_(x3.cuda())
    print(y1_.size(), y3_.size())       # [1, 64, 56, 56]   [1, 64, 56, 56]
    print(torch.all(y1==y1_.cpu()))     # False
    print(torch.all(y3==y3_.cpu()))     # False
    print(y1[0, 0, 0, 0] - y1_.cpu()[0, 0, 0, 0])   # 5.9605e-08 说明cpu和gpu结果略有不同


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
with torch.inference_mode():        # 使用 inference_mode 也依然是训练的输出
    trace_model = torch.jit.script(train_eval)
    y = trace_model(x3)
print(y.size())                     # [1, 64, 56, 56]


#-------------------------------------------#
# 设置eval会使用推理模式导出
train_eval.eval()                   # important, don't forget!!!
trace_model = torch.jit.script(train_eval)
with torch.inference_mode():
    y = trace_model(x3)
print(y.size())                     # [1, 3136, 64]
