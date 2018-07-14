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


def calc_gradient_penalty(args, model, real_data, gen_data):
    batch_size = args.batch_size
    if args.dataset == 'mnist':
        if args.size == '1x':
            if args.layer == 'conv1':
                datashape = (3, 3, 32)
            if args.layer == 'conv2':
                datashape = (3, 3, 64)
        if args.size == 'wide':
            if args.layer == 'conv1':
                datashape = (3, 3, 128, 1)
            if args.layer == 'conv2':
                datashape = (3, 3, 256, 128)
        if args.size == 'wide7':
            if args.layer == 'conv1':
                datashape = (128, 1, 7, 7)
            if args.layer == 'conv2':
                datashape = (128, 256, 7, 7)
    elif args.dataset == 'cifar':
        if args.size in ['presnet', 'resnet']:
            datashape = (3, 3, 512)
        if args.size == '1x':
            datashape = (3, 3, 128)

    # alpha = torch.rand(batch_size, 1)
    alpha = torch.rand(datashape[0], 1)
    # if args.layer == 'conv1':
    #     alpha = alpha.expand(real_data.size()).cuda()
    # alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size))
    # alpha = alpha.contiguous().view(batch_size, *(datashape[::-1])).cuda()
    alpha = alpha.expand(datashape[0], int(real_data.nelement()/datashape[0]))
    alpha = alpha.contiguous().view(*datashape).cuda()
    interpolates = alpha * real_data + ((1 - alpha) * gen_data).cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates = model(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * args.gp
    return gradient_penalty


def gen_layer(args, netG, data):
    init = []
    for i in range(data.shape[1]):
        gen_params = netG(data)
        init.append(gen_params)
    g = torch.stack(init)
    return g
 

def clf_loss(args, iter, sample):
    """ get the classifier loss on the generated samples """
    sample = sample.transpose(1, 0)
    acc, loss = utils.test_samples(args, iter, sample)
    return acc, loss * args.beta
    

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


