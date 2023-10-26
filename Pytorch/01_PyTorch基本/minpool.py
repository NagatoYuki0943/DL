import torch


a = torch.tensor([[[
    [1.,  2,  5,  6],
    [ 3,  4,  7,  8],
    [ 9, 10, 13, 14],
    [11, 12, 15, 16],
]]])

pool = torch.nn.MaxPool2d(2)
res1 = pool(a)
print(res1)
# tensor([[[[ 4.,  8.],
#           [12., 16.]]]])


# minpool 通过  -maxpool(-x) 来实现
a_ = -a
res2 = -pool(a_)
print(res2)
# tensor([[[[ 1.,  5.],
#           [ 9., 13.]]]])
