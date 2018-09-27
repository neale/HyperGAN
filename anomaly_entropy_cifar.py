import matplotlib
matplotlib.use('Agg')
import os
import sys
import argparse
import natsort
import numpy as np
from glob import glob
from math import e, log
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import mode
from scipy.stats import entropy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image

import utils
import netdef
import datagen
import attacks
import models.cifar_clf as models
import models.models_cifar_small as hyper

import logging
import foolbox
from foolbox.criteria import Misclassification
foolbox_logger = logging.getLogger('foolbox')
logging.disable(logging.CRITICAL);
import warnings
warnings.filterwarnings("ignore")

import statsmodels.api as sm

def load_args():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='')
    parser.add_argument('--net', type=str, default='mednet', metavar='N', help='')
    parser.add_argument('--dataset', type=str, default='cifar', metavar='N', help='')
    parser.add_argument('--mdir', type=str, default='models/', metavar='N', help='')
    parser.add_argument('--scratch', type=bool, default=False, metavar='N', help='')
    parser.add_argument('--ft', type=bool, default=False, metavar='N', help='')
    parser.add_argument('--hyper', type=bool, default=False, metavar='N', help='')
    parser.add_argument('--task', type=str, default='train', metavar='N', help='')
    parser.add_argument('--exp', type=str, default='0', metavar='N', help='')

    args = parser.parse_args()
    return args


def plot_empirical_cdf(args, a, n):
    a = np.asarray(a).reshape(-1)
    np.save('/scratch/eecs-share/ratzlafn/{}_{}_ent_data_{}.npy'.format(
        args.dataset, n, args.exp), a)
    ecdf = sm.distributions.ECDF(a)
    x = np.linspace(min(a), max(a))
    y = ecdf(x)
    np.save('cifarhidden_cdfx.npy', x)
    np.save('cifar_hidden_cdfy.npy', y)
    plt.plot(x, y)
    plt.grid(True)
    print ('ecdf with {} samples'.format(len(a)))
    plt.savefig('/scratch/eecs-share/ratzlafn/{}_{}_ent_{}.png'.format(
        args.dataset, n, args.exp))


# Basically I need to modify an attack so that it takes the
# gradient under a new network for each iteration: IFGSM
# One step attacks, do they transfer, no need to modify attacks here. 
def sample_fmodel(args, hypernet, arch):
    w_batch = utils.sample_hypernet_cifar(hypernet)
    rand = np.random.randint(32)
    sample_w = (w_batch[0][rand],
                w_batch[1][rand],
                w_batch[2][rand],
                w_batch[3][rand],
                w_batch[4][rand])
    model = utils.weights_to_clf(sample_w, arch, args.stat['layer_names'])
    return model


def unnormalize(x):
    x *= .3081
    x += .1307
    return x


def normalize(x):
    x -= .1307
    x /= .3081
    return x


class FusedNet(nn.Module):
    def __init__(self, networks):
        super(FusedNet, self).__init__()
        self.networks = networks

    def forward(self, x):
        logits = []
        for network in self.networks:
            logits.append(network(x))
        logits = torch.stack(logits).mean(0)
        return logits


def run_anomaly_cifar(args, hypernet):
    arch = get_network(args)
    cls = [5, 6, 7, 8, 9]
    rcls = [0, 1, 2, 3, 4]
    train, test  = datagen.load_cifar_hidden(args, rcls)
    _vars, _stds, _ents = [], [], []
    model = sample_fmodel(args, hypernet, arch) 
    for n_models in [10, 100, 500]:
        _ents, _ents1, _ents2, _ents3 = [], [], [], []
        for idx, (data, target) in enumerate(train):
            data, target = data.cuda(), target.cuda()
            pred_labels = []
            logits = []
            softs = []
            with torch.no_grad():
                for _ in range(n_models):
                    model = sample_fmodel(args, hypernet, arch) 
                    output = model(data)
                    softs.append(F.softmax(output, 1))
            _softs = torch.stack(softs).float()
            std1 = _softs.std(0)
            std2, std3, std4 = std1*2, std1*3, std1*4
            mean_softmax = _softs.mean(0)
            ent0 = float(entropy((mean_softmax).transpose(0, 1).detach()).mean())
            #ent1 = float(entropy((mean_softmax + std1).transpose(0, 1).detach()).mean())
            #ent2 = float(entropy((mean_softmax + std2).transpose(0, 1).detach()).mean())
            #ent3 = float(entropy((mean_softmax + std3).transpose(0, 1).detach()).mean())
            #ent4 = float(entropy((mean_softmax + std4).transpose(0, 1).detach()).mean())

            _ents.append(ent0)
            #_ents.append(ent1)
            #_ents.append(ent2)
            #_ents.append(ent3)
            #_ents.append(ent4)
            if idx > 100: 
                break;
        plot_empirical_cdf(args, _ents, n_models)
        print ('mean ent: {}'.format(torch.tensor(_ents).mean()))

 
def run_anomaly_model(args, models):
    arch = get_network(args)
    model = FusedNet(models)
    cls = [5, 6, 7, 8, 9]
    rcls = [0, 1, 2, 3, 4]
    train, test  = datagen.load_cifar_hidden(args, cls)
    _vars, _stds, _ents = [], [], []
    _ents, _ents1, _ents2, _ents3 = [], [], [], []
    for idx, (data, target) in enumerate(train):
        data, target = data.cuda(), target.cuda()
        pred_labels = []
        logits = []
        softs = []
        output = model(data)
        soft = F.softmax(output, 1)
        ent = float(entropy(soft.transpose(0, 1).detach()).mean())

        _ents.append(ent)
        if idx > 100: 
            break;
    plot_empirical_cdf(args, _ents, len(models))
    print ('mean ent: {}'.format(torch.tensor(_ents).mean()))

     
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
    model = models.MedNet().cuda()
    return model



def run_anomaly(args, path):
    if args.hyper:
        paths = glob(path + '/*.pt')
        path = [x for x in paths if 'hypermnist_0_0.984390625.pt' in x][0]
        path = './hypercifar_hidden_0.75561875.pt'
        path ='./hypercifar_hidden_0.72089375.pt'
        #path = './hypercifar_hidden_0.70113125.pt'
        path = './hypercifar_hidden_0.85234375.pt'
        hypernet = utils.load_hypernet_cifar(path)
        run_anomaly_cifar(args, hypernet)
    else:
        paths = glob('cifar_clf*')
        models = []
        for path in paths:
            model = get_network(args)
            model.load_state_dict(torch.load(path)['state_dict'])
            models.append(model.eval())
        run_anomaly_model(args, models)

if __name__ == '__main__':
    args = load_args()
    args.stat = netdef.nets()[args.net]
    args.shapes = netdef.nets()[args.net]['shapes']
    if args.scratch:
        path = '/scratch/eecs-share/ratzlafn/HyperGAN/'
        if args.hyper:
            path = path +'exp_models'
    else:
        path = './'

    if args.task == 'anomaly':
        run_anomaly(args, path)
    else:
        raise NotImplementedError
