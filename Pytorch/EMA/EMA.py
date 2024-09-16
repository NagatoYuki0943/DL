import matplotlib.pyplot as plt
import torch
from torch import nn
from copy import deepcopy

torch.manual_seed(0)


class EMA:
    def __init__(self, beta: float = 0.995):
        assert beta > 0 and beta < 1
        self.beta = beta

    def update(self, old, new):
        if old is None:
            return new
        return self.beta * old + (1 - self.beta) * new


def test_ema():
    ema = EMA()
    ema_model = None
    models = range(1, 100)
    ema_models = []
    for model in models:
        ema_model = ema.update(ema_model, model)
        ema_models.append(ema_model)
        print(ema_model)
    plt.plot(models, ema_models)
    plt.grid(True)
    plt.xlabel("models")
    plt.ylabel("ema_models")
    plt.show()


class EMAModel(nn.Module):
    def __init__(self, ma_model: nn.Module = None, beta: float = 0.995) -> None:
        super().__init__()
        assert beta > 0 and beta < 1
        self.beta = beta
        self.ma_model = ma_model
        if self.ma_model is not None:
            self.no_requires_grad()

    @property
    def model(self):
        return self.ma_model

    def no_requires_grad(self):
        """将ma_model梯度设置为False"""
        for parameter in self.ma_model.parameters():
            parameter.requires_grad_(False)

    def update_moving_average(self, old, new):
        return self.beta * old + (1 - self.beta) * new

    def update(self, current_model: nn.Module):
        if self.ma_model is None:
            self.ma_model = deepcopy(current_model)
            self.no_requires_grad()
        else:
            with torch.no_grad():  # 非必须使用 no_grad
                for ma_parameters, current_parameters in zip(
                    self.ma_model.parameters(), current_model.parameters()
                ):
                    ma_weight, current_weight = (
                        ma_parameters.data,
                        current_parameters.data,
                    )
                    ma_parameters.data = self.update_moving_average(
                        ma_weight, current_weight
                    )


# 初始化ema_model
def test_mea_model1():
    model = nn.Linear(1, 1).cuda(0)
    ema_model = EMAModel(ma_model=deepcopy(model), beta=0.6)

    # 查看权重
    def print_state_dict():
        print(dict(model.state_dict()))
        print(dict(ema_model.model.state_dict()))
        print()

    # 查看梯度
    def show_requires_grad():
        for parameters in model.parameters():
            print(parameters.requires_grad)
        for parameters in ema_model.model.parameters():
            print(parameters.requires_grad)
        print()

    print_state_dict()
    # {'weight': tensor([[-0.0075]], device='cuda:0'), 'bias': tensor([0.5364], device='cuda:0')}
    # {'weight': tensor([[-0.0075]], device='cuda:0'), 'bias': tensor([0.5364], device='cuda:0')}
    show_requires_grad()
    # True
    # True
    # False
    # False

    # 更新模型参数
    for parameters in model.parameters():
        parameters.data = parameters.data * 2
    print_state_dict()
    # {'weight': tensor([[-0.0150]], device='cuda:0'), 'bias': tensor([1.0729], device='cuda:0')}
    # {'weight': tensor([[-0.0075]], device='cuda:0'), 'bias': tensor([0.5364], device='cuda:0')}

    # 更新ema_model
    ema_model.update(model)
    print_state_dict()
    # {'weight': tensor([[-0.0150]], device='cuda:0'), 'bias': tensor([1.0729], device='cuda:0')}
    # {'weight': tensor([[-0.0105]], device='cuda:0'), 'bias': tensor([0.7510], device='cuda:0')}  ema_model权重更新了
    show_requires_grad()
    # True
    # True
    # False
    # False


# 不初始化ema_model
def test_mea_model2():
    model = nn.Linear(1, 1).cuda(0)
    ema_model = EMAModel(beta=0.6)

    # 查看权重
    def print_state_dict():
        print(dict(model.state_dict()))
        print(dict(ema_model.model.state_dict()))
        print()

    # 查看梯度
    def show_requires_grad():
        for parameters in model.parameters():
            print(parameters.requires_grad)
        for parameters in ema_model.model.parameters():
            print(parameters.requires_grad)
        print()

    # 更新ema_model
    ema_model.update(model)
    print_state_dict()
    # {'weight': tensor([[-0.8230]], device='cuda:0'), 'bias': tensor([-0.7359], device='cuda:0')}
    # {'weight': tensor([[-0.8230]], device='cuda:0'), 'bias': tensor([-0.7359], device='cuda:0')} 初始化相同的参数
    show_requires_grad()
    # True
    # True
    # False
    # False

    # 更新模型参数
    for parameters in model.parameters():
        parameters.data = parameters.data * 2
    print_state_dict()
    # {'weight': tensor([[-1.6461]], device='cuda:0'), 'bias': tensor([-1.4719], device='cuda:0')} 权重更新
    # {'weight': tensor([[-0.8230]], device='cuda:0'), 'bias': tensor([-0.7359], device='cuda:0')}

    # 更新ema_model
    ema_model.update(model)
    print_state_dict()
    # {'weight': tensor([[-1.6461]], device='cuda:0'), 'bias': tensor([-1.4719], device='cuda:0')}
    # {'weight': tensor([[-1.1523]], device='cuda:0'), 'bias': tensor([-1.0303], device='cuda:0')} 更新ema_model权重
    show_requires_grad()
    # True
    # True
    # False
    # False


if __name__ == "__main__":
    test_ema()
    print("*" * 100)
    test_mea_model1()
    print("*" * 100)
    test_mea_model2()
