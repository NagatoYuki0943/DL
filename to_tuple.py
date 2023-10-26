""" https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/helpers.py
Layer/Module Helpers

Hacked together by / Copyright 2020 Ross Wightman
"""
from itertools import repeat
import collections.abc


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


print(to_1tuple(3))     # (3,)
print(to_2tuple(3))     # (3, 3)
print(to_3tuple(3))     # (3, 3, 3)
print(to_4tuple(3))     # (3, 3, 3, 3)
to_5tuple = to_ntuple(5)
print(to_5tuple(3))     # (3, 3, 3, 3, 3)
