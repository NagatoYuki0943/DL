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
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

from flax import linen as nn
import jax
import jax.numpy as jnp

ModuleDef = Any


class ResNetBlock(nn.Module):
    """ResNet block."""

    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(
        self,
        x,
    ):
        residual = x
        y = self.conv(self.filters, (3, 3), self.strides)(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3))(y)
        y = self.norm(scale_init=nn.initializers.zeros_init())(y)

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
    """ResNetV1.5."""

    stage_sizes: Sequence[int]
    block_cls: ModuleDef
    num_classes: int
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: Callable = nn.relu
    conv: ModuleDef = nn.Conv

    @nn.compact
    def __call__(self, x, train: bool = True):
        conv = partial(self.conv, use_bias=False, dtype=self.dtype)
        norm = partial(
            nn.BatchNorm,
            use_running_average=not train,
            momentum=0.9,
            epsilon=1e-5,
            dtype=self.dtype,
            # axis_name='batch',
        )

        x = conv(
            self.num_filters,
            (7, 7),
            (2, 2),
            padding=[(3, 3), (3, 3)],
            name="conv_init",
        )(x)
        # x = norm(name='bn_init')(x)
        x = norm()(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(
                    self.num_filters * 2**i,
                    strides=strides,
                    conv=conv,
                    norm=norm,
                    act=self.act,
                )(x)
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
        x = jnp.asarray(x, self.dtype)
        return x


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


# 定义模型的前向传播函数
def predict(model: nn.Module, variables: dict, input_sample: jax.Array) -> jax.Array:
    # 使用模型的apply方法进行前向传播
    outputs = model.apply(variables, input_sample, mutable=["batch_stats"])
    return outputs


if __name__ == "__main__":
    rng: jax.Array = jax.random.key(0)
    model = ResNet18(num_classes=10, dtype=jnp.float32)
    input_sample: jax.Array = jnp.ones((2, 64, 64, 3), jnp.float32)

    variables = model.init(rng, input_sample)

    # 调用前向传播函数
    outputs = predict(model, variables, input_sample)
    print(len(outputs))  # 2

    print(outputs[0].shape)  # (2, 10)

    for key, value in outputs[1].items():
        print(key)  # batch_stats
        for k, v in value.items():
            print(f"--> {k}")
            # --> BatchNorm_0
            # --> ResNetBlock_0
            # --> ResNetBlock_1
            # --> ResNetBlock_2
            # --> ResNetBlock_3
            # --> ResNetBlock_4
            # --> ResNetBlock_5
            # --> ResNetBlock_6
            # --> ResNetBlock_7
