import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
import matplotlib.pyplot as plt
import natsort
import utils
import datagen
import argparse
from glob import glob
import models.cifar_clf as models


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
    parser.add_argument('--net', type=str, default='mednet', metavar='N',
            help='network to train [ctiny, wide, wide7, cnet, mednet]')
    parser.add_argument('--data', type=str, default='cifar', help='')
    parser.add_argument('--mdir', type=str, default='models/', help='')


    args = parser.parse_args()
    return args


def train(model, grad=False, e=2):

    train_loss, train_acc = 0., 0.
    train_loader, _ = datagen.load_data(None)
    criterion = nn.CrossEntropyLoss()
    """
    for child in list(model.children())[:-1]:
        # print ('removing {}'.format(child))
        for param in child.parameters():
            param.requires_grad = False
    """
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
    for epoch in range(1):
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
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            acc = 100. * correct / total
            utils.progress_bar(i, len(train_loader),
                    'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss/(i+1), 100.*correct/total, correct, total))
        acc, loss = test(model, epoch)
    return acc, loss


def test(model, epoch=None, grad=False):
    model.eval()
    _, test_loader = datagen.load_data(None)
    test_loss = 0.
    correct = 0.
    criterion = nn.CrossEntropyLoss()
    for i, (data, target) in enumerate(test_loader):
        data, target = Variable(data).cuda(), Variable(target).cuda()
        output = model(data)
        if grad is False:
            test_loss += criterion(output, target).item()
        else:
            test_loss += criterion(output, target)
        pred = output.data.max(1, keepdim=True)[1]
        output = None
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
    test_loss /= len(test_loader.dataset)
    acc = correct.item() / len(test_loader.dataset)
    if epoch:
        print ('Epoch: {}, Average loss: {}, Accuracy: {}/{} ({}%)'.format(
            epoch, test_loss, correct, len(test_loader.dataset),
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
        model = models.CNet().cuda()
    elif args.net == 'wide':
        model = models.WideNet().cuda()
    elif args.net == 'ctiny':
        model = models.CTiny().cuda()
    elif args.net == 'lenet':
        model = models.LeNet().cuda()
    elif args.net == 'mednet':
        model = models.MedNet().cuda()
    else:
        raise NotImplementedError
    return model


def run_model_search(args):
    for i in range(0, 2):
        print ("\nRunning CIFAR Model {}...".format(i))
        model = get_network(args)
        # model = w_init(model, 'normal')
        print (model)
        acc = train(model)
        extract_weights_all(args, model, i)
        torch.save(model.state_dict(),
                args.mdir+'cifar/{}/cifar_model_{}'.format(args.net, i))


""" Load a batch of networks to extract weights """
def load_models(args):

    model = get_network(args)
    #paths = glob(args.mdir+'cifar/{}/*.pt'.format(args.net))
    paths = glob('exp_models/*.pt'.format(args.net))
    natpaths = natsort.natsorted(paths)
    ckpts = []
    print (len(paths))
    for i, path in enumerate(natpaths):
        print ("loading model {}".format(path))
        ckpt = torch.load(path)
        ckpts.append(ckpt)
        #model.load_state_dict(ckpt)
        model.load_state_dict(ckpt['state'])
        acc, loss = test(model, 0)
        train(model, 0)
        print ('Acc: {}, Loss: {}'.format(acc, loss))
        #extract_weights_all(args, model, i)


if __name__ == '__main__':
    args = load_args()
    load_models(args)
    #run_model_search(args)

