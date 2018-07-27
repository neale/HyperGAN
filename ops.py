import torch
import natsort
import datagen
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import torch.autograd as autograd

from glob import glob
from scipy.misc import imsave
import train_mnist as mnist
import train_cifar as cifar
from presnet import PreActResNet18
from resnet import ResNet18
import os
import sys
import time
import math

import utils
import torch.nn as nn
import torch.nn.init as init


param_dir = './params/sampled/mnist/test1/'


def gen_layer(args, netG, data):
    init = []
    shape = args.shapes[args.id]
    if len(shape) == 2:  # Linear layer, handle differently
        # Everything should be a multiple of the first smallest layer 
        iters = shape[0]*shape[1] // args.gcd
    else:
        iters = shape[0] // args.batch_size
    for i in range(iters):
        gen_params = netG(data)
        init.append(gen_params)
    g = torch.stack(init)
    g = g.view(shape)
    return g
 

def clf_loss(args, sample):
    """ get the classifier loss on the generated samples """
    # sample = sample.transpose(1, 0)
    return utils.test_samples(args, sample)
    

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)
