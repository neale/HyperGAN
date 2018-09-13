import torch.nn as nn
import torch.nn.functional as F


class WideNet(nn.Module):
    def __init__(self):
        super(WideNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 640, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(640)
        self.conv2 = nn.Conv2d(640, 1280, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(1280)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1280*8*8, 1280)
        self.fc2 = nn.Linear(1280, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool(x)
        x = x.view(-1, 1280*8*8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNet(nn.Module):
    def __init__(self):
        super(CNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128*8*8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool(x)
        x = x.view(-1, 128*8*8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MedNet(nn.Module):
    def __init__(self):
        super(MedNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, bias=False)
        self.conv2 = nn.Conv2d(16, 32, 3, bias=False)
        self.conv3 = nn.Conv2d(32, 32, 3, bias=False)
        self.fc1   = nn.Linear(128, 64, bias=False)
        self.fc2   = nn.Linear(64, 10, bias=False)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv3(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class CTiny(nn.Module):
    def __init__(self):
        super(CTiny, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.dropout1 = nn.Dropout2d(.1)
        self.conv2 = nn.Conv2d(32, 64, 5, stride=2)
        self.dropout2 = nn.Dropout2d(.2)
        self.linear1 = nn.Linear(169*64, 128)
        self.dropout3 = nn.Dropout2d(.3)
        self.linear2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1) 
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.linear1   = nn.Linear(16*5*5, 120)
        self.linear2   = nn.Linear(120, 84)
        self.linear3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.linear1(out))
        out = F.relu(self.linear2(out))
        out = self.linear3(out)
        return out
