import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import presnet
import natsort
import utils
from glob import glob

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True,
                                        transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True,
                                          num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True,
                                       transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=2)


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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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


class TinyNet(nn.Module):
    def __init__(self):
        super(TinyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 7, padding=2)
        self.conv2 = nn.Conv2d(8, 16, 7, padding=4)
        self.conv3 = nn.Conv2d(16, 16, 7, padding=2)
        self.conv4 = nn.Conv2d(16, 32, 7, padding=4)
        self.conv5 = nn.Conv2d(32, 32, 7, padding=2)
        self.conv6 = nn.Conv2d(32, 64, 7, padding=2)
        self.conv7 = nn.Conv2d(64, 64, 7, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*2*2, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = self.pool(x)
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = self.pool(x)
        x = F.leaky_relu(self.conv5(x))
        x = F.leaky_relu(self.conv6(x))
        x = F.leaky_relu(self.conv7(x))
        # print (x.size())
        x = x.view(-1, 64*2*2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train(model):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    for epoch in range(400):
        scheduler.step()
        model.train()
        acc = 0
        total = 0
        correct = 0
        train_loss = 0
        print ("Epoch {}\n".format(epoch))
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            acc = 100. * correct / total
            utils.progress_bar(i, len(trainloader),
                         'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(i+1), 100.*correct/total, correct, total))
    return acc


def test(model):
    model.eval()
    correct = 0
    total = 0
    for (images, labels) in testloader:
        images, labels = Variable(images).cuda(), labels.cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    acc = float(correct) / total
    return acc


def extract_weights(model, id):
    state = model.state_dict()
    cnames = [x for x in list(state.keys()) if 'conv' in x]
    names = [x for x in cnames if 'weight' in x]
    print (names)
    for name in names:
        dir = name[:5]
        conv = state[name]
        params = conv.cpu().numpy()
        # utils.plot_histogram(params, save=True)
        # return
        save_dir = 'params/cifar/resnet18/{}/'.format(dir)
        print ("making ", save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print ('saving param size: ', params.shape)
        np.save('./params/cifar/resnet18/{}/cifar_params_{}_{}'.format(dir, dir, id), params)


def w_init(model, dist='normal'):
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            if dist == 'normal':
                nn.init.normal(layer.weight.data)
            if dist == 'uniform':
                nn.init.kaiming_uniform(layer.weight.data)
    return model


def run_model_search():
    for i in range(1200, 1300):
        print ("\nRunning CIFAR Model {}...".format(i))
        model = presnet.PreActResNet18().cuda()
        # model = w_init(model, 'normal')
        acc = train(model)
        extract_weights(model, i)
        torch.save(model.state_dict(),
                   './models/cifar/resnet18/cifar_{}_{}.pt'.format(i, acc))


def load_models():
    model = Net().cuda()
    paths = glob('./models/cifar/*.pt')
    natpaths = natsort.natsorted(paths)
    for i, path in enumerate(natpaths):
        print ("loading model {}".format(path))
        ckpt = torch.load(path)
        model.load_state_dict(ckpt)
        test_new(model)
        import sys
        sys.exit(0)
        for k in range(10):
            test(model, k)
        # extract_weights(model, i)


if __name__ == '__main__':
    run_model_search()
    load_models()

