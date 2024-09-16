"""
torch.cuda.memory_allocated(device)     来获取当前进程中Torch.Tensor所占用的GPU显存占用字节数。

torch.cuda.max_memory_allocated(device) 告诉你到调用函数为止所达到的最大的GPU显存占用字节数。

torch.cuda.memory_reserved(device)      查看特定设备上缓存分配器当前所占用的GPU显存

torch.cuda.max_memory_reserved(device)  查看特定设备上缓存分配器最大所占用的GPU显存

"""

import torch

# 模型初始化
linear1 = torch.nn.Linear(1024, 1024, bias=False).cuda()
print(torch.cuda.memory_allocated(0) / 1024 / 1024, "MB")  # 4.0 MB

linear2 = torch.nn.Linear(1024, 1, bias=False).cuda()
print(torch.cuda.memory_allocated(0) / 1024 / 1024, "MB")  # 4.00390625 MB

# 输入定义
inputs = torch.tensor([[1.0] * 1024] * 1024).cuda()
print(torch.cuda.memory_allocated(0) / 1024 / 1024, "MB")  # 8.00390625 MB

# 前向传播
loss = sum(linear2(linear1(inputs)))
print(torch.cuda.memory_allocated(0) / 1024 / 1024, "MB")  # 20.12939453125 MB

# 后向传播
loss.backward()
print(torch.cuda.memory_allocated(0) / 1024 / 1024, "MB")  # 28.25830078125 MB

# 再来一次~
loss = sum(linear2(linear1(inputs)))
print(torch.cuda.memory_allocated(0) / 1024 / 1024, "MB")  # 32.25830078125 MB

loss.backward()
print(torch.cuda.memory_allocated(0) / 1024 / 1024, "MB")  # 28.25830078125 MB

print(torch.cuda.max_memory_allocated(0) / 1024 / 1024, "MB")  # 36.2666015625 MB

# 查看特定设备上缓存分配器当前所占用的GPU显存
print(torch.cuda.memory_reserved(0) / 1024 / 1024, "MB")  # 42.0 MB

# 查看特定设备上缓存分配器最大所占用的GPU显存
print(torch.cuda.max_memory_reserved(0) / 1024 / 1024, "MB")  # 42.0 MB
