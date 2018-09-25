import torch.nn as nn
import torch.nn.functional as F


""" net with divisible parameters """
class Small(nn.Module):
    def __init__(self):
        super(Small, self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(1, 64, 7, 1, 4, bias=False),
                nn.ReLU(True),
                nn.MaxPool2d(4, 4),
                )
        self.linear = nn.Linear(3136, 10, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 3136)
        x = self.linear(x)
        return x

""" net with divisible parameters """
class Small2(nn.Module):
    def __init__(self):
        super(Small2, self).__init__()
        self.lnames = None
        self.conv1 = nn.Sequential(
                nn.Conv2d(1, 32, 5, stride=1, bias=False),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(32, 32, 5, stride=1, bias=False),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                )
        self.linear = nn.Linear(512, 10, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 512)
        x = self.linear(x)
        return x

""" net with divisible parameters """
class Small2_MC(nn.Module):
    def __init__(self):
        super(Small2_MC, self).__init__()
        self.lnames = None
        self.conv1 = nn.Sequential(
                nn.Conv2d(1, 32, 5, stride=1, bias=False),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(.5),
                nn.MaxPool2d(2, 2),
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(32, 32, 5, stride=1, bias=False),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(.5),
                nn.MaxPool2d(2, 2),
                )
        self.linear1 = nn.Linear(512, 128, bias=False)
        self.dropout = nn.Dropout(.5)
        self.linear2 = nn.Linear(128, 10, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 512)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
