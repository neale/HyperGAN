from __future__ import print_function
import os
import argparse
import natsort
import numpy as np
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import utils

mdir = '/data0/models/HyperGAN/models/'
# Training settings
def load_args():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--net', type=str, default='wide', metavar='N',
                        help='network to train [tiny, wide, wide7]')


    args = parser.parse_args()
    return args


def load_data():
    torch.cuda.manual_seed(1)
    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=32, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=32, shuffle=True, **kwargs)
    return train_loader, test_loader

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
        x = self.fc1(x)
        x = self.fc2(x)
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
        self.fc1 = nn.Sequential(
                nn.Linear(2*2*256, 1024), 
                nn.ReLU(True),
                )
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 2*2*256)
        x = self.fc1(x)
        x = self.fc2(x)
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
        self.fc1 = nn.Sequential(
                nn.Linear(2*2*64, 128), 
                nn.ReLU(True),
                )
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 2*2*64)
        x = self.fc1(x)
        x = self.fc2(x)
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
        self.fc1 = nn.Sequential(
                nn.Linear(2*2*64, 128), 
                nn.ReLU(True),
                )
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 2*2*64)
        x = self.fc1(x)
        x = self.fc2(x)
        return x



def train(model):
    
    args = load_args()
    train_loader, _ = load_data()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    for epoch in range(10):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data).cuda(), Variable(target).cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        acc = test(model, epoch)
    return acc


def test(model):
    model.eval()
    _, test_loader = load_data()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    for data, target in test_loader:
        data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda()
        output = model(data)
        test_loss += criterion(output, target).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)
    print('Average loss: {}, Accuracy: {}/{} ({}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return acc


""" Extract weights of a loaded model """
def extract_weights(model, id):
    state = model.state_dict()
    conv2 = state['conv1.0.weight']
    params = conv2.cpu().numpy()
    utils.plot_histogram(params, save=True)
    print ('saving param size: ', params.shape)
    np.save('./params/mnist/wide7/conv1/mnist_params_conv1_{}'.format(id), params)


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
        save_dir = 'params/mnist/{}/{}/'.format(args.net, l)
        if not os.path.exists(save_dir):
            print ("making ", save_dir)
            os.makedirs(save_dir)
        print ('saving param size: ', params.shape)
        np.save('./params/mnist/{}/{}/{}_{}'.format(args.net, l, l, id), params)


""" 
init all weights in the net from a normal distribution
Does not work for ResNets 
"""
def w_init(model, dist='normal'):
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            if dist == 'normal':
                nn.init.normal(layer.weight.data)
            if dist == 'uniform':
                nn.init.kaiming_uniform(layer.weight.data)
    return model


""" returns instance of specific model without weights """
def get_network(args):
    if args.net == 'net':
        model = Net().cuda()
    elif args.net == 'wide':
        model = WideNet().cuda()
    elif args.net == 'wide7':
        model = WideNet7().cuda()
    elif args.net == 'tiny':
        model = TinyNet().cuda()
    else:
        raise NotImplementedError
    return model


""" train and save models and their weights """
def run_model_search(args):

    for i in range(0, 1000):
        print ("\nRunning MNIST Model {}...".format(i))
        model = get_network(args)
        print (model)
        model = w_init(model, 'uniform')
        acc = train(model)
        extract_weights(model, i)
        torch.save(model.state_dict(),
                './models/mnist/wide7/mnist_model_{}_{}.pt'.format(i, acc))


""" Load a batch of networks to extract weights """
def load_models(args):
   
    model = get_network(args)
    paths = glob('./models/mnist/{}/*.pt'.format(args.net))
    natpaths = natsort.natsorted(paths)
    for i, path in enumerate(natpaths):
        print ("loading model {}".format(path))
        ckpt = torch.load(path)
        model.load_state_dict(ckpt)
        #test(model, 0)
        extract_weights_all(args, model, i)


if __name__ == '__main__':
    args = load_args()
    
    load_models(args)
    run_model_search(args)
