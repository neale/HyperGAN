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


def batch_rbf(x, y, h_min=1e-3):
    """
        x (tensor): A tensor of shape (Nx, B, D) containing Nx particles
        y (tensor): A tensor of shape (Ny, B, D) containing Ny particles
        h_min(`float`): Minimum bandwidth.
    """
    Nx, Bx, Dx = x.shape 
    Ny, By, Dy = y.shape
    assert Bx == By
    assert Dx == Dy
    diff = x.unsqueeze(1) - y.unsqueeze(0) # Nx x Ny x B x D
    dist_sq = torch.sum(diff**2, -1).mean(dim=-1) # Nx x Ny
    values, _ = torch.topk(dist_sq.view(-1), k=dist_sq.nelement()//2+1)
    median_sq = values[-1]
    h = median_sq / np.log(Nx)
    h = torch.max(h, torch.tensor([h_min]).cuda())
    # Nx x Ny
    kappa = torch.exp(-dist_sq / h)
    # Nx x Ny x B x D
    kappa_grad = torch.einsum('ij,ijkl->ijkl', kappa, -2 * diff / h)

    return kappa, kappa_grad


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

