"""
torch.cdist(x1, x2, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
Computes batched the p-norm distance between each pair of the two collections of row vectors.
计算两个行向量集合的每对之间的 p 范数距离。
"""

import torch

# [a,c]
a = torch.tensor([[0.0, 0.0], [3.0, 3.0]])

# [b,c]
b = torch.tensor([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])


# [a,b]
res1 = torch.cdist(
    a, b, p=2
)  # 代表a中每个点到b中每个点的距离，每一行代表a中一个点到b中全部点的距离
print(res1)
# tensor([[1.4142, 2.2361, 3.1623],     \sqrt 2,  \sqrt 5, \sqrt 10
#         [2.8284, 2.2361, 2.0000]])    2\sqrt 2, \sqrt 5, 2
v, i = torch.topk(res1, k=2)  # topk返回值和下标
print(v)
# tensor([[3.1623, 2.2361],
#         [2.8284, 2.2361]])
print(i)
# tensor([[2, 1],
#         [0, 1]])
print("*" * 100)


# [b,a] 相当于[a,b]的转置，距离不变
res2 = torch.cdist(
    b, a, p=2
)  # 代表b中每个点到a中每个点的距离，每一行代表b中一个点到a中全部点的距离
print(res2)
# tensor([[1.4142, 2.8284],             \sqrt 2,  2\sqrt 2
#         [2.2361, 2.2361],             \sqrt 5,  \sqrt 5
#         [3.1623, 2.0000]])            \sqrt 10, 2
v, i = torch.topk(res2, k=1)  # topk返回值和下标
print(v)
# tensor([[2.8284],
#         [2.2361],
#         [3.1623]])
print(i)
# tensor([[1],
#         [0],
#         [0]])
