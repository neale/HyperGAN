import matplotlib
matplotlib.use('Agg')
import os
import sys
import argparse
import natsort
import numpy as np
from math import e, log
from glob import glob
import matplotlib.pyplot as plt
import statsmodels.api as sm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import models.mnist_clf as models
import models.models_mnist_small as hyper

import utils
import datagen
import netdef


def load_args():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='')
    parser.add_argument('--net', type=str, default='small2', metavar='N', help='')
    parser.add_argument('--dataset', type=str, default='mnist', metavar='N', help='')
    parser.add_argument('--mdir', type=str, default='models/', metavar='N', help='')
    parser.add_argument('--scratch', type=bool, default=False, metavar='N', help='')
    parser.add_argument('--ft', type=bool, default=False, metavar='N', help='')
    parser.add_argument('--hyper', type=bool, default=False, metavar='N', help='')
    parser.add_argument('--task', type=str, default='train', metavar='N', help='')

    args = parser.parse_args()
    return args


def plot_empirical_cdf(args, a):
    a = np.asarray(a).reshape(-1)
    np.save('/scratch/eecs-share/ratzlafn/{}_ent_data.npy'.format(args.dataset), a)
    ecdf = sm.distributions.ECDF(a)
    x = np.linspace(min(a), max(a))
    y = ecdf(x)
    plt.step(x, y)
    print ('ecdf with {} samples'.format(len(a)))
    plt.savefig('/scratch/eecs-share/ratzlafn/{}_ent.png'.format(args.dataset))


def entropy(y, base=None):
    if len(y) <= 1:
        return 0
    value, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    n_classes = np.count_nonzero(probs)
    if n_classes <= 1:
        return 0
    entropy = 0.
    base = e if base is None else base
    for i in probs:
        entropy -= i * log(i, base)
    return entropy


def sample_model(hypernet, arch):
    w_batch = utils.sample_hypernet(hypernet)
    rand = np.random.randint(32)
    sample_w = (w_batch[0][rand], w_batch[1][rand], w_batch[2][rand])
    model = utils.weights_to_clf(sample_w, arch, args.stat['layer_names'])
    model.eval()
    return model


def run_anomoly_omni(args, hypernet):
    arch = get_network(args)
    omni_loader = datagen.load_omniglot(args)
    _vars, _stds, _ents = [], [], []
    model = sample_model(hypernet, arch) 
    for idx, (data, target) in enumerate(omni_loader):
        data, target = data.cuda(), target.cuda()
        pred_labels = []
        for _ in range(1000):
            model = sample_model(hypernet, arch) 
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            pred_labels.append(pred.view(pred.numel()))
            print (pred_labels)

        p_labels = torch.stack(pred_labels).float().transpose(0, 1)
        _vars.append(p_labels.var(1).mean())
        _stds.append(p_labels.std(1).mean())
        _ents.append(np.apply_along_axis(entropy, 1, p_labels))

    plot_empirical_cdf(args, _ents)
    print ('mean var: {}, min var: {}, max var:{}, std: {}'.format(
        torch.tensor(_vars).mean(), torch.tensor(_vars).max(), torch.tensor(_vars).min(),
        torch.tensor(_stds).mean()))


def run_anomoly_notmnist(args, hypernet):
    arch = get_network(args)
    train, test  = datagen.load_notmnist(args)
    _vars, _stds, _ents = [], [], []
    model = sample_model(hypernet, arch) 
    for idx, (data, target) in enumerate(train):
        data, target = data.cuda(), target.cuda()
        pred_labels = []
        for _ in range(1000):
            model = sample_model(hypernet, arch) 
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            pred_labels.append(pred.view(pred.numel()))

        p_labels = torch.stack(pred_labels).float().transpose(0, 1)
        _vars.append(p_labels.var(1).mean())
        _stds.append(p_labels.std(1).mean())
        _ents.append(np.apply_along_axis(entropy, 1, p_labels))
        
    plot_empirical_cdf(args, _ents)
    print ('mean var: {}, max var: {}, min var:{}, std: {}'.format(
        torch.tensor(_vars).mean(), torch.tensor(_vars).max(), torch.tensor(_vars).min(),
        torch.tensor(_stds).mean()))


def run_anomoly_mnist(args, hypernet):
    arch = get_network(args)
    train, test  = datagen.load_mnist(args)
    _vars, _stds, _ents = [], [], []
    model = sample_model(hypernet, arch) 
    for idx, (data, target) in enumerate(test):
        data, target = data.cuda(), target.cuda()
        pred_labels = []
        for _ in range(10):
            model = sample_model(hypernet, arch) 
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            pred_labels.append(pred.view(pred.numel()))

        p_labels = torch.stack(pred_labels).float().transpose(0, 1)
        _vars.append(p_labels.var(1).mean())
        _stds.append(p_labels.std(1).mean())
        _ents.append(np.apply_along_axis(entropy, 1, p_labels))
    
    plot_empirical_cdf(args, _ents)
    print ('mean var: {}, max var: {}, min var:{}, std: {}'.format(
        torch.tensor(_vars).mean(), torch.tensor(_vars).max(), torch.tensor(_vars).min(),
        torch.tensor(_stds).mean()))


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
        model = models.Net().cuda()
    elif args.net == 'wide':
        model = models.WideNet().cuda()
    elif args.net == 'wide7':
        model = models.WideNet7().cuda()
    elif args.net == 'tiny':
        model = models.TinyNet().cuda()
    elif args.net == 'fcn':
        model = models.FCN().cuda()
    elif args.net == 'fcn2':
        model = models.FCN2().cuda()
    elif args.net == 'small':
        model = models.Small().cuda()
    elif args.net == 'small2':
        model = models.Small2().cuda()
    else:
        raise NotImplementedError
    return model


def run_anomaly(args, path):
    if args.hyper:
        paths = glob(path + '/*.pt')
        path = [x for x in paths if 'hypermnist_0_0.984390625.pt' in x][0]
        hypernet = utils.load_hypernet(path)
        if args.dataset == 'omniglot':
            run_anomoly_omni(args, hypernet)
        elif args.dataset == 'notmnist':
            run_anomoly_notmnist(args, hypernet)
        elif args.dataset == 'mnist':
            run_anomoly_mnist(args, hypernet)
        else:
            raise NotImplementedError
    else:
        model = get_network(args)
        path = 'mnist_clf.pt'
        model.load_state_dict(torch.load(path))
        run_adv_model(args, model)
    

""" train and save models and their weights """
def run_model_search(args, path):

    for i in range(0, 500):
        print ("\nRunning MNIST Model {}...".format(i))
        model = get_network(args)
        print (model)
        model = w_init(model, 'normal')
        acc, loss, model = train(args, model)
        #extract_weights_all(args, model, i)
        torch.save(model.state_dict(),
                mdir+'mnist/{}/mnist_model_{}_{}.pt'.format(args.net, i, acc))


""" Load a batch of networks to extract weights """
def load_models(args, path):
   
    model = get_network(args)
    paths = glob(path + '*.pt')
    print (path)
    paths = [path for path in paths if 'mnist' in path]
    natpaths = natsort.natsorted(paths)
    accs = []
    losses = []
    natpaths = [x for x in natpaths if 'hypermnist_mi_0.987465625' in x]
    for i, path in enumerate(natpaths):
        print ("loading model {}".format(path))
        if args.hyper:
            hn = utils.load_hypernet(path)
            for i in range(10):
                samples = utils.sample_hypernet(hn)
                print ('sampled a batches of {} networks'.format(len(samples[0])))
                for i, sample in enumerate(zip(samples[0], samples[1], samples[2])):
                    model = utils.weights_to_clf(sample, model, args.stat['layer_names'])
                    acc, loss = test(args, model)
                    print (i, ': Test Acc: {}, Loss: {}'.format(acc, loss))
                    accs.append(acc)
                    losses.append(loss)
                    #acc, loss = train(args, model)
                    #print ('Test1 Acc: {}, Loss: {}'.format(acc, loss))
                    #extract_weights_all(args, model, i)
            print(accs, losses)
        else:
            ckpt = torch.load(path)
            state = ckpt['state_dict']
            try:
                model.load_state_dict()
            except RuntimeError:
                model_dict = model.state_dict()
                filtered = {k:v for k, v in state.items() if k in model_dict}
                model_dict.update(filtered)
                model.load_state_dict(filtered)



if __name__ == '__main__':
    args = load_args()
    args.stat = netdef.nets()[args.net]
    args.shapes = netdef.nets()[args.net]['shapes']
    if args.scratch:
        path = '/scratch/eecs-share/ratzlafn/HyperGAN/'
        if args.hyper:
            path = path +'exp_models'
    else:
        path = mdir+'mnist/{}/'.format(args.net)

    if args.task == 'test':
        load_models(args, path)
    elif args.task =='train':
        run_model_search(args, path)
    elif args.task == 'anomaly':
        run_anomaly(args, path)
    else:
        raise NotImplementedError
