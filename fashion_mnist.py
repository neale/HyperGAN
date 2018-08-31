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

# Training settings
def load_args():
    parser = argparse.ArgumentParser(description='PyTorch Fashion MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--net', type=str, default='small2', metavar='N',
                        help='network to train [tiny, wide, wide7, fcn]')
    parser.add_argument('--mdir', type=str, default='/data0/models/HyperGAN/models/')
    parser.add_argument('--data', type=str, default='fashion_mnist')

    args = parser.parse_args()
    return args


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


def train(model, grad=False, ft=False):
    
    train_loss, train_acc = 0., 0.
    train_loader, _ = load_data()
    criterion = nn.CrossEntropyLoss()
    if ft:
        for child in list(model.children())[:2]:
            print('removing {}'.format(child))
            for param in child.parameters():
                param.requires_grad = False
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    for epoch in range(1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        acc, loss = test(model, epoch)
    return acc, loss


def test(model, epoch=None, grad=False):
    model.eval()
    _, test_loader = load_data()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        output = model(data)
        if grad is False:
            test_loss += criterion(output, target).item() # sum up batch loss
        else:
            test_loss += criterion(output, target)

        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)
    if epoch is not None:
        print('Average loss: {}, Accuracy: {}/{} ({}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return acc, test_loss


def extract_weights_all(args, model, id):
    state = model.state_dict()
    conv_names = [x for x in list(state.keys()) if 'conv' in x]
    fc_names = [x for x in list(state.keys()) if 'linear' in x]
    cnames = [x for x in conv_names if 'weight' in x] 
    fnames = [x for x in fc_names if 'weight' in x]
    names = cnames + fnames
    print (names)
    for i, name in enumerate(names):
        print (name)
        l = name[:-7]
        conv = state[name]
        params = conv.cpu().numpy()
        save_dir = 'params/{}/{}/{}/'.format(args.data, args.net, l)
        if not os.path.exists(save_dir):
            print ("making ", save_dir)
            os.makedirs(save_dir)
        print ('saving param size: ', params.shape)
        np.save('./params/{}/{}/{}/{}_{}'.format(args.data, args.net, l, l, id), params)


def measure_models(m1, m2, i):
    m1_conv = m1['conv1.0.weight'].view(-1)
    m2_conv = m2['conv1.0.weight'].view(-1)
    m1_linear = m1['linear.weight']
    m2_linear = m2['linear.weight']
    
    print (m1_conv.shape, m1_linear.mean(0).shape)
    
    l1_conv = (m1_conv - m2_conv).abs().sum()
    l1_linear = (m1_linear - m2_linear).abs().sum()
    print ("\nL1 Dist: {} - {}".format(l1_conv, l1_linear))

    l2_conv = np.linalg.norm(m1_conv - m2_conv)
    l2_linear = np.linalg.norm(m1_linear - m2_linear)
    print ("L2 Dist: {} - {}".format(l2_conv, l2_linear))

    linf_conv = np.max((m1_conv-m2_conv).abs().cpu().numpy())
    linf_linear = np.max((m1_linear-m2_linear).abs().cpu().numpy())
    print ("Linf Dist: {} - {}".format(linf_conv, linf_linear))

    cov_m1_m2_conv = np.cov(m1_conv, m2_conv)[0, 1]
    cov_m1_m2_linear = np.cov(m1_linear, m2_linear)[0, 1]
    print ("Cov m1-m2: {} - {}".format(cov_m1_m2_conv, cov_m1_m2_linear))

    cov_m1_conv_linear = np.cov(m1_conv, m1_linear.mean(0))[0, 1]
    cov_m2_conv_linear = np.cov(m2_conv, m2_linear.mean(0))[0, 1]
    print ("Cov m1-conv-linear: {} - {}\n\n".format(cov_m1_conv_linear, cov_m2_conv_linear))
   
    return
    utils.plot_histogram([m1_conv.view(-1).cpu().numpy(), m2_conv.view(-1).cpu().numpy()],
            save=False)
    utils.plot_histogram([m1_linear.view(-1).cpu().numpy(), m2_linear.view(-1).cpu().numpy()],
            save=False)
    
    for l, name in [(m1_conv, 'conv1.0'), (m1_linear, 'linear')]:
        params = l.cpu().numpy()
        save_dir = 'params/{}/{}/{}'.format(args.data, args.net, name)
        if not os.path.exists(save_dir):
            print ("making ", save_dir)
            os.makedirs(save_dir)
        path = '{}/{}_{}.npy'.format(save_dir, name, i)
        print (i)
        print ('saving param size: ', params.shape, 'to ', path)
        np.save(path, params)


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
    if args.net == 'small':
        model = Small().cuda()
    elif args.net == 'small2':
        model = Small2().cuda()
    else:
        raise NotImplementedError
    return model


""" train and save models and their weights """
def run_model_search(args):

    for i in range(0, 500):
        print ("\nRunning {} Model {}...".format(args.data, i))
        model = get_network(args)
        print (model)
        model = w_init(model, 'normal')
        acc = train(model)
        #extract_weights_all(args, model, i)
        #torch.save(model.state_dict(),
        #        mdir+'mnist/{}/mnist_model_{}_{}.pt'.format(args.net, i, acc))


""" Load a batch of networks to extract weights """
def load_models(args):
   
    model = get_network(args)
    #paths = glob(args.mdir+args.data+'/{}/*.pt'.format(args.net))
    paths = glob('exp_models/*.pt'.format(args.net))
    natpaths = natsort.natsorted(paths)
    ckpts = []
    print (len(paths))
    for i, path in enumerate(natpaths):
        print ("loading model {}".format(path))
        ckpt = torch.load(path)
        ckpts.append(ckpt)
        model.load_state_dict(ckpt['state'])
        #model.load_state_dict(ckpt)
        test(model, 0)
        train(model, ft=True)
        #extract_weights_all(args, model, i)
    """
    for i in range(len(ckpts)):
        for j in range(i, len(ckpts)):
            model1 = ckpts[i]
            model2 = ckpts[j]
            measure_models(model1, model2, i)
    """

if __name__ == '__main__':
    args = load_args()
    
    #run_model_search(args)
    load_models(args)
