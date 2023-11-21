# 对比

比喻成一摞书，这摞书总共有 N 本，每本有 C 页，每页有 H 行，每行 有W 个字符。

![NB LN IN GN](BN LN IN GN.assets/NB LN IN GN(1803.08494 Group Normalization).png)

1. BN是在batch上，对 `N、H、W` 做归一化，而保留通道 `C` 的维度。BN 相当于把这些书按页码一一对应地加起来，再除以每个页码下的字符总数：`N×H×W`。

2. LN在通道方向上，对 `C、H、W` 归一化，而保留通道 `B` 的维度。LN 相当于把每一本书的所有字加起来，再除以这本书的字符总数：`C×H×W`。

3. IN在图像像素上，对 `H、W` 做归一化，而保留通道 `B、C` 的维度。IN 相当于把一页书中所有字加起来，再除以该页的总字数：`H×W`。

4. GN将channel分组，然后再做归一化。GN 相当于把一本 `C` 页的书平均分成 `G` 份，每份成为有 `C/G` 页的小册子，对每个小册子做Norm: `C/G×H×W`。

    1. 如果我们将组号设置为 `G = 1`，则GN变为LN 。LN假设层中的所有通道都做出“类似的贡献”。GN比LN受限制更少，**因为假设每组通道（而不是所有通道）都受共享均值和方差的影响; 该模型仍然具有为每个群体学习不同分布的灵活性**。这导致GN相对于LN的代表能力提高。*而且pytorch中LN要求数据维度*

        > LN对于4维数据在最后3维上处理,要把 `CHW` 都写进参数, 对于NLP的3维,会在最后的dim维度上处理
        >
        > 实际应用中一般不知道图片的 `HW`，因此使用LN不方便，使用GN(g=1)更方便
    
        ```python
        import torch
        from torch import nn
        
        class GroupNorm1(nn.GroupNorm):
            """ Group Normalization with 1 group, equivalent with LayerNorm.
            Input: tensor in shape [B, C, *]
            """
        
            def __init__(self, num_channels, **kwargs):
                #             将 num_groups 数设置为1
                super().__init__(1, num_channels, **kwargs)
        
        
        # GN vs LN example:
        B, C, H, W = 1, 3, 224, 224
        x = torch.randn(B, C, H, W)
        
        gn = nn.GroupNorm(1, C)                         # 分为1组(等价LN),通道为3,数据是4维的
        print(gn(x).size())                             # [1, 3, 224, 224]
        
        
        ln = nn.LayerNorm([C, H, W])                    # LN对于4维数据在最后3维上处理,要把 CHW 都写进参数, 对于NLP的3维,会在最后的dim维度上处理
        print(ln(x).size())                             # [1, 3, 224, 224]
        
        # 实际使用LN处理图片一般会把图片的形状转换为mlp的形状 [batch, position, channel],将channel调至最后,在channel上计算LN,计算完再转换回来形状
        ln = nn.LayerNorm(C)
        x = x.reshape(B, C, -1).transpose(1, 2)         # [1, 3, 224, 224] -> [1, 3, 224*224] -> [1, 224*224, 3]
        y = ln(x)
        y = y.transpose(1, 2).reshape(B, C, H, W)       # [1, 224*224, 3] -> [1, 3, 224*224] -> [1, 3, 224, 224]
        print(y.size())                                 # [1, 3, 224, 224]
        
        
        # mlp序列处理实例
        x = torch.randn(1, 196, 768)
        ln = nn.LayerNorm(768)                          # 处理最后的dim维度
        print(ln(x).size())                             # [1, 196, 768]
        ```
    
    2. 如果我们将组号设置为G = C（即每组一个通道），则GN变为IN。 但是**IN只能依靠空间维度来计算均值和方差，并且错过了利用信道依赖的机会**。
    
    3. GN与LN和IN有关，这两种标准化方法在**训练循环（RNN / LSTM）或生成（GAN）模型**方面特别成功

另外，还需要注意它们的映射参数 γ 和 β 的区别：对于 BN，IN，GN， 其 γ 和 β 都是维度等于通道数 C 的向量。而对于 LN，其 γ 和 β 都是维度等于 normalized_shape 的矩阵。

最后，BN 和 IN 可以设置参数：`momentum`和`track_running_stats`来获得在整体数据上更准确的均值和标准差。LN 和 GN 只能计算当前 batch 内数据的真实均值和标准差。
