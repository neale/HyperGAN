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
    parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='')
    parser.add_argument('--net', type=str, default='mednet', metavar='N', help='')
    parser.add_argument('--dataset', type=str, default='cifar', help='')
    parser.add_argument('--mdir', type=str, default='models/', help='')
    parser.add_argument('--scratch', type=bool, default=False, help='')
    parser.add_argument('--ft', type=bool, default=False, help='')
    parser.add_argument('--hyper', type=bool, default=False, help='')
    parser.add_argument('--task', type=str, default='train', help='')

    args = parser.parse_args()
    return args


def train(args, model, grad=False):
    if args.dataset == 'cifar':
        train_loader, _ = datagen.load_cifar(args)
    elif args.dataset == 'cifar100':
        train_loader, _ = datagen.load_10_class_cifar100(args)

    train_loss, train_acc = 0., 0.
    criterion = nn.CrossEntropyLoss()
    if args.ft:
        for child in list(model.children())[:-1]:
            # print ('removing {}'.format(child))
            for param in child.parameters():
                param.requires_grad = False
    
    optimizer = optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    #optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    for epoch in range(args.epochs):
        model.train()
        total = 0
        correct = 0
        for i, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
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
        acc, loss = test(args, model, epoch+1)
    return acc, loss


def test(args, model, epoch=None, grad=False):
    model.eval()
    if args.dataset == 'cifar':
        _, test_loader = datagen.load_cifar(args)
    elif args.dataset == 'cifar100':
        _, test_loader = datagen.load_10_class_cifar100(args)
    test_loss = 0.
    correct = 0.
    criterion = nn.CrossEntropyLoss()
    for i, (data, target) in enumerate(test_loader):
        data, target = data.cuda(), target.cuda()
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


def run_model_search(args, path):
    for i in range(0, 2):
        print ("\nRunning CIFAR Model {}...".format(i))
        model = get_network(args)
        # model = w_init(model, 'normal')
        print (model)
        acc, loss = train(args, model)
        #extract_weights_all(args, model, i)
        torch.save({'state_dict': model.state_dict()},
                path+'cifar_clf_{}.pt'.format(acc))


""" Load a batch of networks to extract weights """
def load_models(args, path):

    model = get_network(args)
    paths = glob(path + '*.pt'.format(args.net))
    paths = natsort.natsorted(paths)
    for i, path in enumerate(paths):
        print ("loading model {}".format(path))
        ckpt = torch.load(path)
        state = ckpt['state_dict']
        try: #bias issue: TODO remove this bit after HyperNet retraining
            model.load_state_dict(state)
        except RuntimeError:
            model_dict = model.state_dict()
            filtered = {k: v for k, v in state.items() if k in model_dict}
            model_dict.update(filtered)
            model.load_state_dict(filtered)

        acc, loss = test(args, model, 0)
        print ('Test0 Acc: {}, Loss: {}'.format(acc, loss))
        acc, loss = train(args, model)
        print ('Train Acc: {}, Loss: {}'.format(acc, loss))
        #extract_weights_all(args, model, i)


if __name__ == '__main__':
    args = load_args()
    
    if args.task == 'test':
        if args.hyper:
            path = 'exp_models/'
        else:
            path = args.mdir+'cifar/{}/'.format(args.net)
        if args.scratch:
            path = '/scratch/eecs-share/ratzlafn/HyperGAN/' + path
        load_models(args, path)

    elif args.task == 'train':
        path = args.mdir+'cifar/{}/'.format(args.net)
        if args.scratch:
            path = '/scratch/eecs-share/ratzlafn/HyperGAN/' + path
        
        run_model_search(args, path)
    else:
        raise NotImplementedError

