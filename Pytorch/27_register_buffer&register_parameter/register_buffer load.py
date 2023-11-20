#-------------------------------------------------------------------------------------#
#   self.register_buffer('name', Tensor)的操作，该方法的作用是定义一组参数，
#   该组参数的特别之处在于：模型训练时不会更新（即调用 optimizer.step() 后该组参数不会变化，
#   只可人为地改变它们的值，但是保存模型时，该组参数又作为模型参数不可或缺的一部分被保存。
#-------------------------------------------------------------------------------------#


import torch
from torch import nn, Tensor


def load_dynamic_buffer_module(model: nn.Module, state_dict: dict) -> nn.Module:
    """解决使用register_buffer保存的参数初始化形状和保存权重形状不统一的问题

    refer: https://github.com/openvinotoolkit/anomalib/blob/main/src/anomalib/models/components/base/dynamic_module.py#L32

    Args:
        model (nn.Module): model
        state_dict (dict): state_dict
    """
    model_state_dict = model.state_dict()
    for key in state_dict.keys():
        # 如果对应位置的形状不一样,就修改模型参数的形状
        if model_state_dict[key].shape != state_dict[key].shape:
            attribute: Tensor = getattr(model, key)
            attribute.resize_(state_dict[key].shape)

    model.load_state_dict(state_dict)
    return model


class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.fc = nn.Linear(10, 20)
        self.register_buffer('buffer', torch.Tensor())
        self.buffer: Tensor

    def forward(self, x):
        x = self.fc(x)
        self.buffer = x
        return x


def save():
    model = Model()
    x = torch.ones(1, 10)
    model(x)
    torch.save(model.state_dict(), "register_buffer.pth")


def load():
    model = Model()
    state_dict = torch.load("register_buffer.pth")
    model = load_dynamic_buffer_module(model, state_dict)
    print(model.buffer.size()) # [1, 20]


if __name__ == "__main__":
    save()
    load()
