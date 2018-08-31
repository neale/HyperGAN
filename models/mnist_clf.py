import torch.nn as nn
import torch.nn.functional as F


""" trains a 4x wide MNIST network with 3x3 filters """
class WideNet(nn.Module):
    def __init__(self):
        super(WideNet, self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(1, 128, 3),
                nn.ReLU(True),
                nn.MaxPool2d(2, 2),
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(128, 256, 3),
                nn.ReLU(True),
                nn.MaxPool2d(2, 2),
                nn.MaxPool2d(2, 2),
                )
        self.fc1 = nn.Sequential(
                nn.Linear(2*2*256, 1024), 
                nn.ReLU(True),
                )
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 2*2*256)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


""" 4x Wide MNIST net with 7x7 filters """
class WideNet7(nn.Module):
    def __init__(self):
        super(WideNet7, self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(1, 128, 7),
                nn.ReLU(True),
                nn.MaxPool2d(2, 2),
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(128, 256, 7),
                nn.ReLU(True),
                nn.MaxPool2d(2, 2),
                )
        self.linear1 = nn.Sequential(
                nn.Linear(2*2*256, 1024), 
                nn.ReLU(True),
                )
        self.linear2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 2*2*256)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


""" Standard small MNIST net with 3x3 filters """
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(1, 32, 3),
                nn.ReLU(True),
                nn.MaxPool2d(2, 2),
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(32, 64, 3),
                nn.ReLU(True),
                nn.MaxPool2d(2, 2),
                nn.MaxPool2d(2, 2),
                )
        self.linear1 = nn.Sequential(
                nn.Linear(2*2*64, 128), 
                nn.ReLU(True),
                )
        self.linear2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 2*2*64)
        x = self.linear1(x)
        x = self.linear2(x)
        return x

""" Standard small MNIST net with 3x3 filters """
class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(1, 32, 3),
                nn.ReLU(True),
                nn.MaxPool2d(2, 2),
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(32, 64, 3),
                nn.ReLU(True),
                nn.MaxPool2d(2, 2),
                )
        self.linear = nn.Linear(1600, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 1600)
        x = self.linear(x)
        return x

""" Standard small MNIST net with 3x3 filters """
class FCN2(nn.Module):
    def __init__(self):
        super(FCN2, self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(1, 64, 7),
                nn.ReLU(True),
                nn.MaxPool2d(4, 4),
                )
        self.linear = nn.Linear(1600, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 1600)
        x = self.linear(x)
        return x

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

""" Narrow MNIST net with 3x3 filters """
class TinyNet(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(1, 8, 3),
                nn.ReLU(True),
                nn.MaxPool2d(2, 2),
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(8, 16, 3),
                nn.ReLU(True),
                nn.MaxPool2d(2, 2),
                nn.MaxPool2d(2, 2),
                )
        self.linear1 = nn.Sequential(
                nn.Linear(2*2*64, 128), 
                nn.ReLU(True),
                )
        self.linear2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 2*2*64)
        x = self.linear1(x)
        x = self.linear2(x)
        return x
