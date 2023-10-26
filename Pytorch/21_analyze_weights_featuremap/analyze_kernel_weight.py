#-----------------------------------------------#
#   查看模型权重分布
#-----------------------------------------------#

import torch
from efficientnet_v1_00 import efficientnet_b0
import matplotlib.pyplot as plt
import numpy as np


# 直接使用权重即可,不用实例化也没问题
# create model
model = efficientnet_b0(num_classes=50)
# load model weights
model_weight_path = "./efficientnet_v1_b0_pt_sche_best_model.pkl"
model.load_state_dict(torch.load(model_weight_path))
print(model)

# 获取key,通过key获取数据
weights_keys = model.state_dict().keys()
for key in weights_keys:
    # remove num_batches_tracked para(in bn)
    if "num_batches_tracked" in key:
        continue

    # [out_channel,   in_channel,...]
    # [kernel_number, kernel_channel, kernel_height, kernel_width]
    weight_t = model.state_dict()[key].numpy()

    # read a kernel information
    # k = weight_t[0, :, :, :]

    # calculate mean, std, min, max
    weight_mean = weight_t.mean()
    weight_std = weight_t.std(ddof=1)
    weight_min = weight_t.min()
    weight_max = weight_t.max()
    print("mean is {}, std is {}, min is {}, max is {}".format(weight_mean,
                                                               weight_std,
                                                               weight_max,
                                                               weight_min))

    # plot hist image
    plt.close()
    weight_vec = np.reshape(weight_t, [-1]) # 变为一维向量
    plt.hist(weight_vec, bins=50)           # 直方图 bins=50 将数据从小到大分为50份,统计每个区间个数
    plt.title(key)
    plt.show()
