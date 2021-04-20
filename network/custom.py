import torch
import torch.nn as nn


class Custom_CNN(nn.Module):
    def __init__(self, in_channel=3):
        super(Custom_CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channel, 32, 7, 2, 3, bias=False),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, 1, 2, bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, 1, 2, bias=False),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2, 1, 1),
            nn.ReLU(),
        )
        self.linear = nn.Sequential(
            nn.Linear(12800, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.net(x)
        bsz = x.shape[0]
        x = self.linear(x.view(bsz, -1))
        return x.squeeze(1)

class Linear_cls(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Linear_cls, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x, softmax=False):
        if softmax:
            return self.softmax(self.fc(x))
        else:
            return self.fc(x)