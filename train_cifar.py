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
import resnet
import natsort
import utils
import argparse
from glob import glob


mdir = '/data0/models/HyperGAN/models/'

def load_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
            help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
            help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
            help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
            help='learning rate (default: 0.01)')
    parser.add_argument('--net', type=str, default='ctiny', metavar='N',
            help='network to train [ctiny, wide, wide7, cnet]')


    args = parser.parse_args()
    return args


def load_data():
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
            download=False,
            transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=512,
            shuffle=True,
            num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
            download=False,
            transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=512,
            shuffle=False, num_workers=2)

    return trainloader, testloader


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

"""
class CTiny(nn.Module):
    def __init__(self):
        super(CTiny, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(64*2*2, 128)
        self.linear2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 64*2*2)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
"""
class CTiny(nn.Module):
    def __init__(self):
        super(CTiny, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.bn3 = nn.BatchNorm2d(64)
        self.linear1 = nn.Linear(256, 64)
        self.linear2 = nn.Linear(64, 10)
        self.mpool = nn.MaxPool2d(2, 2)
        self.apool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.mpool(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.mpool(out)
        out = F.relu(self.bn3(self.conv3(out)))
        out = self.apool(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.linear1(out))
        out = self.linear2(out)
        return out


def train(model, grad=False, e=2):

    train_loss, train_acc = 0., 0.
    train_loader, _ = load_data()
    criterion = nn.CrossEntropyLoss()
    """
    for child in list(model.children())[:-1]:
        # print ('removing {}'.format(child))
        for param in child.parameters():
            param.requires_grad = False
    """
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
            lr=0.01, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50000, gamma=0.1)

    for epoch in range(100):
        scheduler.step()
        model.train()
        total = 0
        correct = 0
        for i, (data, target) in enumerate(train_loader):
            data, target = Variable(data).cuda(), Variable(target).cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.data[0]
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += predicted.eq(target.data).cpu().sum()
            acc = 100. * correct / total
            utils.progress_bar(i, len(train_loader),
                    'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss/(i+1), 100.*correct/total, correct, total))
        acc, loss = test(model, epoch)
    return acc, loss


def test(model, epoch=None, grad=False):
    model.eval()
    _, test_loader = load_data()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    for i, (data, target) in enumerate(test_loader):
        data, target = Variable(data).cuda(), Variable(target).cuda()
        output = model(data)
        if grad is False:
            test_loss += criterion(output, target).data[0]
        else:
            test_loss += criterion(output, target)
        pred = output.data.max(1, keepdim=True)[1]
        output = None
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)
    if epoch:
        print ('Average loss: {}, Accuracy: {}/{} ({}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    return acc, test_loss


def extract_weights(model, id):
    state = model.state_dict()
    cnames = [x for x in list(state.keys()) if 'conv' in x]
    names = [x for x in cnames if 'weight' in x]
    print (names)
    for name in names:
        print (name)
        dir = name[:-7]
        conv = state[name]
        params = conv.cpu().numpy()
        # utils.plot_histogram(params, save=True)
        # return
        save_dir = 'params/cifar/resnet/{}/'.format(dir)
        print ("making ", save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print ('saving param size: ', params.shape)
        np.save('./params/cifar/resnet/{}/cifar_params_{}_{}'.format(dir, dir, id), params)


def extract_weights_all(args, model, id):
    state = model.state_dict()
    conv_names = [x for x in list(state.keys()) if 'conv' in x]
    fc_names = [x for x in list(state.keys()) if 'fc' in x]
    cnames = [x for x in conv_names if 'weight' in x]
    fnames = [x for x in fc_names if 'weight' in x]
    names = cnames + fnames
    print (names)
    for i, name in enumerate(names):
        print (name)
        l = name[:-7]
        conv = state[name]
        params = conv.cpu().numpy()
        save_dir = 'params/cifar/{}/{}/'.format(args.net, l)
        if not os.path.exists(save_dir):
            print ("making ", save_dir)
            os.makedirs(save_dir)
        print ('saving param size: ', params.shape)
        np.save('./params/cifar/{}/{}/{}_{}'.format(args.net, l, l, id), params)


def w_init(model, dist='normal'):
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            if dist == 'normal':
                nn.init.normal_(layer.weight.data)
            if dist == 'uniform':
                nn.init.kaiming_uniform(layer.weight.data)
    return model


def get_network(args):
    if args.net == 'cnet':
        model = CNet().cuda()
    elif args.net == 'wide':
        model = WideNet().cuda()
    elif args.net == 'ctiny':
        model = CTiny().cuda()
    else:
        raise NotImplementedError
    return model


def run_model_search(args):
    for i in range(0, 100):
        print ("\nRunning CIFAR Model {}...".format(i))
        model = get_network(args)
        model = w_init(model, 'normal')
        acc = train(model)
        extract_weights_all(model, i)
        torch.save(model.state_dict(),
                mdir+'cifar/{}/cifar_model_{}_{}'.format(args.net, i, acc))


""" Load a batch of networks to extract weights """
def load_models(args):

    model = get_network(args)
    paths = glob(mdir+'cifar/{}/*.pt'.format(args.net))
    natpaths = natsort.natsorted(paths)
    ckpts = []
    print (len(paths))
    for i, path in enumerate(natpaths):
        print ("loading model {}".format(path))
        ckpt = torch.load(path)
        ckpts.append(ckpt)
        model.load_state_dict(ckpt)
        test(model, 0)
        #extract_weights_all(args, model, i)


if __name__ == '__main__':
    args = load_args()
    run_model_search(args)
    load_models(args)

