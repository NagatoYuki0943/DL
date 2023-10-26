[神经网络学习小记录74——Pytorch 设置随机种子Seed来保证训练结果唯一_Bubbliiiing的博客-CSDN博客](https://blog.csdn.net/weixin_44791964/article/details/131622957?spm=1001.2014.3001.5502)

# 为什么每次训练结果不同

1. 随机权重，网络有些部分的权重没有预训练，它的值则是随机初始化的，每次随机初始化不同会导致结果不同。

2. 随机数据增强，一般来讲网络训练会进行数据增强，特别是少量数据的情况下，数据增强一般会随机变化光照、对比度、扭曲等，也会导致结果不同。

3. 随机数据读取，喂入训练数据的顺序也会影响结果。

……

应该还有别的随机值，这里不一一列出，这些随机都很容易影响网络的训练结果。

如果能够固定权重、固定数据增强情况、固定数据读取顺序，网络理论上每一次独立训练的结果都是一样的。

# 什么是随机种子

随机种子（Random Seed）是计算机专业术语。一般计算机的随机数都是伪随机数，以一个真随机数（种子）作为初始条件，然后用一定的算法不停迭代产生随机数。

按照这个理解，我们如果可以设置最初的 **真随机数（种子）**，那么后面出现的随机数将会是固定序列。

以random库为例，我们使用如下的代码，前两次为随机生成，后两次为设置随机数生成器种子后生成。

```python
import random

# 生成随机整数
print("第一次随机生成")
print(random.randint(1,100))
print(random.randint(1,100))

# 生成随机整数
print("第二次随机生成")
print(random.randint(1,100))
print(random.randint(1,100))

# 设置随机数生成器种子
random.seed(11)

# 生成随机整数
print("第一次设定种子后随机生成")
print(random.randint(1,100))
print(random.randint(1,100))

# 重置随机数生成器种子
random.seed(11)

# 生成随机整数
print("第二次设定种子后随机生成")
print(random.randint(1,100))
print(random.randint(1,100))
```

结果如下，前两次随机生成的序列不同，后两次设定种子后随机生成的序列相同：

```
第一次随机生成
66
37
第二次随机生成
93
56
第一次设定种子后随机生成
58
72
第二次设定种子后随机生成
58
72
```

# 训练中设置随机种子

一般训练会用到多个库包含有关random的内容。

在pytorch构建的网络中，一般都是使用下面三个库来获得随机数，我们需要对三个库都设置随机种子：

1. torch库；
2. numpy库；

3. random库。

在这里写了一个函数：

```python
#---------------------------------------------------#
#   设置种子
#---------------------------------------------------#
def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

这里面写到了cuda、cudnn这类gpu才会用到的东西，实测发现cpu版本的pytorch也可以正常运行。

torch.backends.cudnn.deterministic=True	**用于保证CUDA 卷积运算的结果确定。**

torch.backends.cudnn.benchmark=False	**是用于保证数据变化的情况下，减少网络效率的变化。为True的话容易降低网络效率。**

只需要在所有初始化前，调用该seed初始化函数即可。

另外，Pytorch一般使用Dataloader来加载数据，Dataloader一般会使用多worker加载多进程来加载数据，此时我们需要使用Dataloader自带的worker_init_fn函数初始化Dataloader启动的多进程，这样才能保证多进程数据加载时数据的确定性。

```python
#---------------------------------------------------#
#   设置Dataloader的种子
#---------------------------------------------------#
def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
```

