import torch
import torch.nn as nn
import torch.nn.init as init
import torch.distributions.multivariate_normal as N
import torch.distributions.uniform as U
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

import os
import math
import numpy as np
from bisect import bisect_right,bisect_left
import seaborn as sns
import matplotlib.pyplot as plt


def weights_to_clf(weights, model, names):
    state = model.state_dict()
    layers = zip(names, weights)
    for i, (name, params) in enumerate(layers):
        name = name + '.weight'
        loader = state[name]
        state[name] = params.detach()
        model.load_state_dict(state)
    return model


class CyclicCosAnnealingLR(_LRScheduler):
    def __init__(self, optimizer,milestones, eta_min=0, last_epoch=-1):
        self.eta_min = eta_min
        self.milestones=milestones
        super(CyclicCosAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.milestones[-1]:
            return [self.eta_min for base_lr in self.base_lrs]
        idx = bisect_right(self.milestones,self.last_epoch)
        left_barrier = 0 if idx==0 else self.milestones[idx-1]
        right_barrier = self.milestones[idx]
        width = right_barrier - left_barrier
        curr_pos = self.last_epoch- left_barrier 
        return [self.eta_min + (base_lr - self.eta_min) *
               (1 + math.cos(math.pi * curr_pos/ width)) / 2
                for base_lr in self.base_lrs]


def hyperlayer_init(layer, d_in, d_out, is_final=False):
    if is_final:
        relu = 1
    else:
        relu = 2
    fan_out = d_in * d_out * 1.0  # 1.0 variance for standard gaussian input
    w_init = np.sqrt(3 * ((2**relu)/fan_out))
    torch.nn.init.uniform_(layer.weight.data, -w_init, w_init)
    return layer


def print_statistics_hypernetwork(dataset, epoch, loss, best_stats):
    best_acc, best_loss = best_stats
    print ('--------------------------------------')
    print ('{} Train, epoch: {}'.format(dataset, epoch))
    print ('HyperNetwork Loss: {}'.format(loss))
    print ('best test loss: {}'.format(best_loss))
    print ('best test acc: {}'.format(best_acc))
    print ('--------------------------------------')


def plot_density_mnist(x_inliers, x_outliers, ens_size, prefix, epoch=0):
    in_entropy, in_variance = x_inliers
    out_entropy, out_variance = x_outliers
    f, axes = plt.subplots(2, 2, figsize=(22, 16))
    plt.suptitle('{} models MNIST'.format(ens_size))
    plt.subplots_adjust(top=0.85)

    sns.distplot(in_entropy, hist=False, label='MNIST', color='m', ax=axes[0,0])
    sns.distplot(out_entropy, hist=False, label='notMNIST', ax=axes[0,0])
    axes[0, 0].set_xlabel('Entropy')
    axes[0, 0].grid(True)

    sns.distplot(in_variance, hist=False, label='MNIST', color='m', ax=axes[0,1])
    sns.distplot(out_variance, hist=False, label='notMNIST', ax=axes[0,1])
    axes[0, 1].set_xlabel('Variance')
    axes[0, 1].grid(True)

    sns.distplot(out_entropy, hist=True, color='b', ax=axes[1, 0])
    axes[1, 0].set_xlabel('Entropy')
    axes[1, 0].set_title('notMNIST entropy')
    axes[1, 0].grid(True)

    sns.distplot(out_variance, hist=True, color='b', ax=axes[1, 1])
    axes[1, 1].set_xlabel('Variance')
    axes[1, 1].set_title('notMNIST variance')
    axes[1, 1].grid(True)

    axes[0, 0].legend(loc='best')
    axes[0, 1].legend(loc='best')
    plt.tight_layout()

    path = prefix+'/epoch{}_{}models.png'.format(epoch, ens_size)
    os.makedirs(prefix, exist_ok=True)
    
    f.savefig(path)
    plt.close('all')

