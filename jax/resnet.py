# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Flax implementation of ResNet V1.5."""

# See issue #620.
# pytype: disable=wrong-arg-count

from functools import partial
from typing import Any, Callable, Sequence
import time

from flax import linen as nn
import jax
import jax.numpy as jnp

ModuleDef = Any


# -------------------------------------------------------------------------
# Flax 的 nn.Module 只是一个“逻辑定义”，不持有任何权重数据。
# 所有的权重在 init 时生成，在 apply 时作为外部参数传入。
# -------------------------------------------------------------------------


class ResNetBlock(nn.Module):
    """ResNet 基础残差块（对应 ResNet-18/34）。"""

    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: tuple[int, int] = (1, 1)

    @nn.compact  # 关键装饰器：允许你在 __call__ 中直接定义层，而不需要在 setup() 中提前声明
    def __call__(self, x):
        residual = x
        # 第一层卷积
        y = self.conv(self.filters, (3, 3), self.strides)(x)
        y = self.norm()(y)
        y = self.act(y)
        # 第二层卷积
        y = self.conv(self.filters, (3, 3))(y)
        # 这里的 scale_init=zeros_init() 是 ResNet 训练的一个小技巧：
        # 让残差分支初始化为 0，这样模型初期就像一个恒等映射，更容易训练。
        y = self.norm(scale_init=nn.initializers.zeros_init())(y)

        # 如果输入输出维度不一致（例如 Stride=2 时），对 Residual 进行投影映射
        if residual.shape != y.shape:
            residual = self.conv(self.filters, (1, 1), self.strides, name="conv_proj")(
                residual
            )
            residual = self.norm(name="norm_proj")(residual)

        return self.act(residual + y)


class BottleneckResNetBlock(nn.Module):
    """Bottleneck ResNet block."""

    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.conv(self.filters, (1, 1))(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3), self.strides)(y)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters * 4, (1, 1))(y)
        y = self.norm(scale_init=nn.initializers.zeros_init())(y)

        if residual.shape != y.shape:
            residual = self.conv(
                self.filters * 4, (1, 1), self.strides, name="conv_proj"
            )(residual)
            residual = self.norm(name="norm_proj")(residual)

        return self.act(residual + y)


class ResNet(nn.Module):
    """ResNetV1.5 主结构。"""

    stage_sizes: Sequence[int]
    block_cls: ModuleDef
    num_classes: int
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: Callable = nn.relu
    conv: ModuleDef = nn.Conv

    @nn.compact
    def __call__(self, x, train: bool = True):
        # 使用 partial 预设一些通用的层配置，类似于 PyTorch 里的构造函数预设
        conv = partial(self.conv, use_bias=False, dtype=self.dtype)
        norm = partial(
            nn.BatchNorm,
            use_running_average=not train,  # 推理时使用累计的均值/方差
            momentum=0.9,
            epsilon=1e-5,
            dtype=self.dtype,
            # axis_name="batch",  # 单个设备指定出错
        )

        # 1. 初始 7x7 卷积层
        x = conv(
            self.num_filters, (7, 7), (2, 2), padding=[(3, 3), (3, 3)], name="conv_init"
        )(x)
        x = norm(name="bn_init")(x)
        x = nn.relu(x)
        # 2. 最大池化
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")

        # 3. 循环构建 ResNet 的四个 Stage
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                # 只有每个 Stage 的第一个 Block 可能需要下采样 (Stride=2)
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(
                    self.num_filters * 2**i,
                    strides=strides,
                    conv=conv,
                    norm=norm,
                    act=self.act,
                )(x)

        # 4. 全局平均池化 (GAP) 并接入全连接层 (Dense)
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
        return jnp.asarray(x, self.dtype)


ResNet18 = partial(ResNet, stage_sizes=[2, 2, 2, 2], block_cls=ResNetBlock)
ResNet34 = partial(ResNet, stage_sizes=[3, 4, 6, 3], block_cls=ResNetBlock)
ResNet50 = partial(ResNet, stage_sizes=[3, 4, 6, 3], block_cls=BottleneckResNetBlock)
ResNet101 = partial(ResNet, stage_sizes=[3, 4, 23, 3], block_cls=BottleneckResNetBlock)
ResNet152 = partial(ResNet, stage_sizes=[3, 8, 36, 3], block_cls=BottleneckResNetBlock)
ResNet200 = partial(ResNet, stage_sizes=[3, 24, 36, 3], block_cls=BottleneckResNetBlock)


ResNet18Local = partial(
    ResNet, stage_sizes=[2, 2, 2, 2], block_cls=ResNetBlock, conv=nn.ConvLocal
)


# Used for testing only.
_ResNet1 = partial(ResNet, stage_sizes=[1], block_cls=ResNetBlock)
_ResNet1Local = partial(
    ResNet, stage_sizes=[1], block_cls=ResNetBlock, conv=nn.ConvLocal
)


# -------------------------------------------------------------------------
# 推理辅助函数：利用 JAX 的编译特性
# -------------------------------------------------------------------------


def make_predict_fn(model: nn.Module):
    # jax.jit 会将 Python 函数编译为 XLA 加速的高性能代码
    @jax.jit
    def predict_step(variables: dict, x: jax.Array):
        # variables: 包含权重 (params) 和状态 (batch_stats) 的字典
        # mutable=['batch_stats']: 在推理时通常设为 False，
        # 但如果要在推理时也更新统计量或返回它们，需标记为 mutable。
        return model.apply(variables, x, train=False, mutable=["batch_stats"])

    return predict_step


if __name__ == "__main__":
    # 1. 初始化随机数种子 (JAX 的随机性是显式的)
    rng = jax.random.key(0)

    # 2. 实例化模型逻辑
    model = ResNet18(num_classes=10)
    input_data = jnp.ones((2, 224, 224, 3))

    # 3. 初始化权重（相当于 PyTorch 里的层创建+权重初始化）
    # Flax 需要一个 sample input 来自动推导各层的 Shape
    variables = model.init(rng, input_data)

    # 4. 构建编译后的预测函数
    predict = make_predict_fn(model)

    # --- 性能测试 ---
    # 第一次调用：包含编译时间（Tracing + XLA 编译）
    t1 = time.time()
    output, new_vars = predict(variables, input_data)
    t2 = time.time()

    # 第二次调用：纯粹的执行时间（在 GPU 上飞快）
    output, new_vars = predict(variables, input_data)
    t3 = time.time()

    print(f"首次运行时间 (含编译): {t2 - t1:.4f} s")
    print(f"二次运行时间 (纯推理): {t3 - t2:.4f} s")
    # 首次运行时间 (含编译): 0.1901 s
    # 二次运行时间 (纯推理): 0.0010 s

    # 5. 输出结果：new_vars 包含了最新的 BatchNorm 统计量
    print(f"输出维度: {output.shape}")
    # 输出维度: (2, 10)

    print("更新变量:")
    for key, value in new_vars.items():
        print(key)
        # batch_stats
        for k, v in value.items():
            print(f"--> {k}")
    # 更新变量:
    # batch_stats
    # --> ResNetBlock_0
    # --> ResNetBlock_1
    # --> ResNetBlock_2
    # --> ResNetBlock_3
    # --> ResNetBlock_4
    # --> ResNetBlock_5
    # --> ResNetBlock_6
    # --> ResNetBlock_7
    # --> bn_init

