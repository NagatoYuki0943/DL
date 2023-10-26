

```python
import torch

# 随机数种子
torch.manual_seed(1234)             # 为处理器设置随机数种子
torch.cuda.manual_seed(1234)        # 为显卡设置随机数种子
torch.cuda.manual_seed_all(1234)    # 多显卡为所有显卡设置随机数种子
```

