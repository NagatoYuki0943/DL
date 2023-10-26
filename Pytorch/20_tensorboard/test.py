#----------------------------------------------------#
#   tensorboard --logdir=logs --samples_per_pligin=images*50
#   localhost:6006
#----------------------------------------------------#


import numpy as np
import torch
from torch import optim,nn
from torchvision import models, transforms
from torch.utils.tensorboard import SummaryWriter
from utils import plot_class_preds

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# 测试轮数
EPOCH = 10

#----------------------------------------------------#
#   实例化SummaryWriter对象
#   参数是保存的目录
#----------------------------------------------------#
tb_writer = SummaryWriter(log_dir="logs")

# 模型
model = models.convnext_tiny(pretrained=False)
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 5)
model.to(device)

# 学习率衰减
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1E-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH, eta_min=1e-5)

# 定义训练以及预测时的预处理方法
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}


#----------------------------------------------------#
#   保存模型结构图
#----------------------------------------------------#
init_img = torch.zeros(1, 3, 224, 224).to(device)
tb_writer.add_graph(model, init_img)

tags = ["train_loss", "accuracy", "learning_rate"]

loss = np.linspace(1, 10, EPOCH)      # 开始,结束,数量
acc  = np.linspace(1, 100, EPOCH)
# lr   = np.linspace(0.001, 0.1, 100)
# lr   = lr[::-1]
# print(loss)
# print(acc)
# print(lr)


for epoch in range(EPOCH):
    #----------------------------------------------------#
    #   add_scalar:  散点图
    #       参数1: 名字
    #       参数2: y轴  不是tensor,是数字
    #       参数3: x轴  不是tensor,是数字
    #----------------------------------------------------#
    tb_writer.add_scalar(tags[0], loss[epoch], epoch)
    tb_writer.add_scalar(tags[1], acc[epoch],  epoch)
    tb_writer.add_scalar(tags[2], optimizer.state_dict()['param_groups'][0]['lr'],   epoch)

    optimizer.zero_grad()
    optimizer.step()
    # 降低学习率
    scheduler.step()

    #----------------------------------------------------#
    #   add_figure:  图片
    #       参数1: 名字
    #       参数2: y轴  matplotlib.pyplot.figure 一个或者列表
    #       参数3: x轴  不是tensor,是数字
    #----------------------------------------------------#
    fig = plot_class_preds(net=model,
                            images_dir="./plot_img",            # 测试图片路径
                            transform=data_transform["val"],    # 图片预处理
                            num_plot=5,                         # 测试几张
                            device=device)
    if fig is not None:
        tb_writer.add_figure("predictions vs. actuals",
                                figure=fig,
                                global_step=epoch)

    #----------------------------------------------------#
    #   直方图,绘制权重
    #       参数1: 名字
    #       参数2: tensor
    #       参数3: 循环次数  不是tensor,是数字
    #----------------------------------------------------#
    tb_writer.add_histogram(tag="stem/conv",
                            values=model.features[0][0].weight,
                            global_step=epoch)
    tb_writer.add_histogram(tag="stem/ln",
                            values=model.features[0][1].weight,
                            global_step=epoch)