import torch
import torch.nn as nn
import torch.nn.init as init
import torch.distributions.multivariate_normal as N
import torch.distributions.uniform as U
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

import math
import numpy as np
from bisect import bisect_right,bisect_left


def sample_hypernet_mnist(args ,hypernet, num):
    netE, W1, W2, W3 = hypernet
    x_dist = create_d(args.ze)
    z = sample_d(x_dist, num)
    codes = netE(z)
    l1 = W1(codes[0])
    l2 = W2(codes[1])
    l3 = W3(codes[2])
    return l1, l2, l3, codes


def sample_hypernet_cifar(args, hypernet, num):
    netE, W1, W2, W3, W4, W5 = hypernet
    x_dist = create_d(args.ze)
    z = sample_d(x_dist, num)
    codes = netE(z)
    l1 = W1(codes[0])
    l2 = W2(codes[1])
    l3 = W3(codes[2])
    l4 = W4(codes[3])
    l5 = W5(codes[4])
    return l1, l2, l3, l4, l5, codes


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
