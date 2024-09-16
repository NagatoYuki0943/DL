"""
数据集 => 训练集 + 测试集
训练集 => 小训练集 + 验证集

训练的时候使用验证集验证,最后使用测试集测试
可以事实查看模型是否过拟合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

batch_size = 200
learning_rate = 0.01
epochs = 5

# 训练集
train_db = datasets.MNIST(
    "../../../data",
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
)
train_loader = DataLoader(train_db, batch_size=batch_size, shuffle=True)

# 测试集
test_db = datasets.MNIST(
    "../../../data",
    train=False,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
)
test_loader = DataLoader(test_db, batch_size=batch_size, shuffle=True)

print("train:", len(train_db), "test:", len(test_db))


# 将训练集 划分为 训练集和验证集                                 划分数量
train_db, val_db = torch.utils.data.random_split(train_db, [50000, 10000])
print("db1:", len(train_db), "db2:", len(val_db))

# 新的训练集
train_loader = DataLoader(train_db, batch_size=batch_size, shuffle=True)

# 验证集
val_loader = DataLoader(val_db, batch_size=batch_size, shuffle=True)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 10),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.model(x)

        return x


device = torch.device("cuda:0")
net = MLP().to(device)
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss().to(device)

for epoch in range(epochs):
    # 训练
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28 * 28)
        data, target = data.to(device), target.cuda()

        predict = net(data)
        loss = loss_fn(predict, target)

        optimizer.zero_grad()
        loss.backward()
        # print(w1.grad.norm(), w2.grad.norm())
        optimizer.step()

        if batch_idx % 100 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

    # 验证集
    test_loss = 0
    correct = 0
    for data, target in val_loader:
        data = data.view(-1, 28 * 28)
        data, target = data.to(device), target.cuda()
        predict = net(data)
        test_loss += loss_fn(predict, target).item()

        # 预测值和真实值比较
        pred = predict.data.max(1)[1]
        correct += pred.eq(target.data).sum()

    test_loss /= len(val_loader.dataset)
    print(
        "\nVAL set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(val_loader.dataset),
            100.0 * correct / len(val_loader.dataset),
        )
    )


# 测试集
test_loss = 0
correct = 0
for data, target in test_loader:
    data = data.view(-1, 28 * 28)
    data, target = data.to(device), target.cuda()
    predict = net(data)
    test_loss += loss_fn(predict, target).item()

    # 预测值和真实值比较
    pred = predict.data.max(1)[1]
    correct += pred.eq(target.data).sum()

test_loss /= len(test_loader.dataset)
print(
    "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
        test_loss,
        correct,
        len(test_loader.dataset),
        100.0 * correct / len(test_loader.dataset),
    )
)
