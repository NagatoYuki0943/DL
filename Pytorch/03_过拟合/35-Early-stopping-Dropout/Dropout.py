"""
torch.nn.Dropout(0.4),  # 丢失 0.4 数据

tf.nn.dropout(0.4)      # 保留 0.4 数据
"""

import  torch
import  torch.nn as nn



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(784, 200),
            nn.Dropout(0.4),  # 丢失 0.4 数据
            nn.ReLU(True),

            nn.Linear(200, 200),
            nn.Dropout(0.4),
            nn.ReLU(),

            nn.Linear(200, 10)
        )


    def forward(self, input):
        return self.net(input)