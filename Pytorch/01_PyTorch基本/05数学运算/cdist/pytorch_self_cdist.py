import torch
from torch import Tensor
import time


def my_cdist_p2(x1: Tensor, x2: Tensor) -> Tensor:
    """这个函数主要是为了解决torch.cdist导出onnx后,使用其他推理引擎推理onnx内存占用过大的问题
        如果使用torchscript推理则不会有内存占用过大的问题,使用原本的torch.cdist即可
        比torch.cdist更慢,不过导出onnx更快
        dim=3时第1个维度代表batch,大了之后相比torch.cdist更慢
        https://github.com/openvinotoolkit/anomalib/issues/440#issuecomment-1191184221
        https://github.com/pytorch/pytorch/issues/15253#issuecomment-491467128

        只能处理二维矩阵
    Args:
        x1 (Tensor): [x, z]
        x2 (Tensor): [y, z]
    Returns:
        Tensor: [x, y]
    """
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.addmm(
        x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2
    ).add_(x1_norm)
    res = res.clamp_min_(1e-30).sqrt_()
    return res


def my_cdist_p2_v1(x1: Tensor, x2: Tensor) -> Tensor:
    """这个函数主要是为了解决torch.cdist导出onnx后,使用其他推理引擎推理onnx内存占用过大的问题
        如果使用torchscript推理则不会有内存占用过大的问题,使用原本的torch.cdist即可

        可以处理二维或者三维矩阵
    Args:
        x1 (Tensor): [x, z] or [b, x, z]
        x2 (Tensor): [y, z] or [b, y, z]

    Returns:
        Tensor: [x, y] or [b, x, y]
    """
    if x1.dim() == x2.dim() == 2:
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(
            x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2
        ).add_(x1_norm)
        res = res.clamp_min_(1e-30).sqrt_()
    elif x1.dim() == x2.dim() == 3:
        # batch=1 不循环加速
        if x1.size(0) == 1:
            # x1.squeeze_(0)    # 这样在pytorch中和 squeeze(0) 效果相同,但是导出onnx会导致输入维度直接变为2维的
            # x2.squeeze_(0)
            x1 = x1.squeeze(0)  # [1, a, x] -> [a, x]
            x2 = x2.squeeze(0)  # [1, b, x] -> [b, x]
            x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
            x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
            res = torch.addmm(
                x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2
            ).add_(x1_norm)
            res = res.clamp_min_(1e-30).sqrt_()
            res.unsqueeze_(0)  # [a, b] -> [1, a, b]
        else:
            # batch > 1
            res = []
            for x1_, x2_ in zip(x1, x2):
                x1_norm = x1_.pow(2).sum(dim=-1, keepdim=True)  # [a, x]
                x2_norm = x2_.pow(2).sum(dim=-1, keepdim=True)  # [a, x]
                res_ = torch.addmm(
                    x2_norm.transpose(-2, -1), x1_, x2_.transpose(-2, -1), alpha=-2
                ).add_(x1_norm)
                res_ = res_.clamp_min_(1e-30).sqrt_()
                res.append(res_)
            res = torch.stack(res, dim=0)  # [a, x] -> [b, a, x]
    return res


def my_cdist_p2_v2(x1: Tensor, x2: Tensor) -> Tensor:
    """这个函数主要是为了解决torch.cdist导出onnx后,使用其他推理引擎推理onnx内存占用过大的问题
        如果使用torchscript推理则不会有内存占用过大的问题,使用原本的torch.cdist即可

        可以处理多维矩阵
    Args:
        x1 (Tensor):[..., x, z]
        x2 (Tensor):[..., y, z]

    Returns:
        Tensor:[..., x, y]
    """
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    # res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
    res = x2_norm.transpose(-2, -1) - 2 * x1 @ x2.transpose(-2, -1) + x1_norm
    res = res.clamp_min_(1e-30).sqrt_()
    return res


def fast_cdist(x1: Tensor, x2: Tensor) -> Tensor:
    """https://github.com/pytorch/pytorch/pull/25799#issuecomment-529021810
        比my_cdist_p2更快
        可以处理多维矩阵
    Args:
        x1 (Tensor):[..., x, z]
        x2 (Tensor):[..., y, z]
    Returns:
        Tensor:[..., x, y]
    """
    adjustment = x1.mean(-2, keepdim=True)
    x1 = x1 - adjustment
    x2 = (
        x2 - adjustment
    )  # x1 and x2 should be identical in all dims except -2 at this point

    # Compute squared distance matrix using quadratic expansion
    # But be clever and do it with a single matmul call
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x1_pad = torch.ones_like(x1_norm)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    x2_pad = torch.ones_like(x2_norm)
    x1_ = torch.cat([-2.0 * x1, x1_norm, x1_pad], dim=-1)
    x2_ = torch.cat([x2, x2_pad, x2_norm], dim=-1)
    res = x1_.matmul(x2_.transpose(-2, -1))

    # Zero out negative values
    res.clamp_min_(1e-30).sqrt_()
    return res


def onnx_cdist(x1: Tensor, x2: Tensor, p=2) -> Tensor:
    """处理二维数据
    Custom cdists function for ONNX export since neither cdists nor
    linalg.norm is currently support by the current PyTorch version 1.10.

    Note that the implementation above does not support p=0.
    Furthermore it is much slower an needs much more memory than PyTorch's native cdist implementation.
    https://github.com/pytorch/pytorch/issues/69201
    """
    s_x1 = torch.unsqueeze(
        x1, dim=1
    )  # [3, 2] -> [3, 1, 2]   onnx就是这样实现的,这样扩展会造成大量内存占用
    s_x2 = torch.unsqueeze(x2, dim=0)  # [4, 2] -> [1, 4, 2]
    # 求差
    diffs = s_x1 - s_x2  # [3, 1, 2] - [1, 4, 2] = [3, 4, 2]

    if p == 1:
        # 差的绝对值求和
        return diffs.abs().sum(dim=2)  # [3, 4, 2] -> [3, 4]
    elif p == "inf":
        # 差的绝对值的最大值
        return diffs.abs().max(dim=2).values  # [3, 4, 2] -> [3, 4]
    elif p % 2 == 0:
        # 差的平方和再开方
        return diffs.pow(p).sum(dim=2).pow(1 / p)  # [3, 4, 2] -> [3, 4]
    else:
        # 差的绝对值的平方和再开方
        return diffs.abs().pow(p).sum(dim=2).pow(1 / p)  # [3, 4, 2] -> [3, 4]


def onnx_cdist_for(x1: Tensor, x2: Tensor, p=2) -> Tensor:
    """处理二维数据
    使用for循环求差,内存占用低,不过仍然不快
    和onnx_cdist对比过结果,结果完全相同
    """
    diffs = torch.zeros(x1.size(0), x2.size(0))  # [3, 4]

    if p == 1:
        for i in range(x1.size(0)):
            for j in range(x2.size(0)):
                diffs[i, j] = (x1[i] - x2[j]).abs().sum(dim=0)
    elif p == "inf":
        for i in range(x1.size(0)):
            for j in range(x2.size(0)):
                diffs[i, j] = (x1[i] - x2[j]).abs().max(dim=0).values
    elif p % 2 == 0:
        for i in range(x1.size(0)):
            for j in range(x2.size(0)):
                diffs[i, j] = (x1[i] - x2[j]).pow(p).sum(dim=0).pow(1 / p)
    else:
        for i in range(x1.size(0)):
            for j in range(x2.size(0)):
                diffs[i, j] = (x1[i] - x2[j]).abs().pow(p).sum(dim=0).pow(1 / p)
    return diffs


if __name__ == "__main__":
    #####################################################
    # 2维对比结果
    x = torch.tensor(
        [
            [0, 0],
            [3, 3.0],
            [4, 4.0],
        ]
    )

    y = torch.tensor(
        [
            [1, 0],
            [1, 1],
            [0, 1.0],
            [2, 2.0],
        ]
    )

    z = torch.cdist(x, y)
    print(z.size())  # [3, 4]
    print(z)
    # tensor([[1.0000, 1.4142, 1.0000, 2.8284],
    #         [3.6056, 2.8284, 3.6056, 1.4142],
    #         [5.0000, 4.2426, 5.0000, 2.8284]])
    z1 = onnx_cdist(x, y, p=2)
    print(torch.all(z == z1))  # True
    z2 = onnx_cdist_for(x, y, p=2)
    print(torch.all(z == z2))  # True
    z3 = my_cdist_p2_v1(x, y)
    print(torch.all(z == z3))  # True
    z4 = my_cdist_p2_v1(x, y)
    print(torch.all(z == z4))  # True
    z5 = fast_cdist(x, y)
    print(torch.all(z == z5))  # False

    #####################################################
    # 三维对比结果
    x = torch.randn(2, 2, 4)
    y = torch.randn(2, 3, 4)

    z = torch.cdist(x, y)
    print(z.size())  # [2, 3, 5]
    z1 = my_cdist_p2_v1(x, y)
    print(z1.size())  # [2, 3, 5]
    print(torch.all(z == z1))  # False
    z2 = my_cdist_p2_v1(x, y)
    print(torch.all(z == z2))  # False
    z3 = fast_cdist(x, y)
    print(torch.all(z == z3))  # False
    print(z.eq(z2).sum().item())  # 7
    print(z.eq(z3).sum().item())  # 7
