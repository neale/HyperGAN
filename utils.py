import os
import sys
import time
import torch
import natsort
import datagen
import argparse
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import itertools
import cv2
import numpy as np

from glob import glob
from scipy.misc import imsave
import torch.nn as nn
import torch.nn.init as init
import torch.distributions.multivariate_normal as N


def sample_z(args, grad=True):
    z = torch.randn(args.batch_size, args.dim, requires_grad=grad).cuda()
    return z


def create_d(shape):
    mean = torch.zeros(shape)
    cov = torch.eye(shape)
    D = N.MultivariateNormal(mean, cov)
    return D


def sample_d(D, shape, scale=1., grad=True):
    z = scale * D.sample((shape,)).cuda()
    z.requires_grad = grad
    return z


def sample_z_like(shape, scale=1., grad=True):
    return torch.randn(*shape, requires_grad=grad).cuda()


def save_model(args, model, optim):
    path = '{}/{}/{}_{}.pt'.format(
            args.dataset, args.model, model.name, args.exp)
    path = model_dir + path
    torch.save({
        'state_dict': model.state_dict(),
        'optimizer': optim.state_dict(),
        'best_acc': args.best_acc,
        'best_loss': args.best_loss
        }, path)


def load_model(args, model, optim):
    path = '{}/{}/{}_{}.pt'.format(
            args.dataset, args.model, model.name, args.exp)
    path = model_dir + path
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['state_dict'])
    optim.load_state_dict(ckpt['optimizer'])
    acc = ckpt['best_acc']
    loss = ckpt['best_loss']
    return model, optim, (acc, loss)


def get_net_only(model):
    net_dict = {
            'state_dict': model.state_dict(),
    }
    return net_dict


def load_net_only(model, d):
    model.load_state_dict(d['state_dict'])
    return model


# this is also hard coded right now to the specific model
# dont sue me 
def save_clf(args, Z, acc):
    """ gross """
    if args.dataset == 'mnist':
        import models.mnist_clf as models
        model = models.Small2().cuda()
    elif args.dataset == 'cifar':
        import models.cifar_clf as models
        model = models.MedNet().cuda() 
    """ end gross """

    state = model.state_dict()
    layers = zip(args.stat['layer_names'], Z)
    for i, (name, params) in enumerate(layers):
        name = name + '.weight'
        loader = state[name]
        state[name] = params.detach()
        assert state[name].equal(loader) == False
        model.load_state_dict(state)
    #import cifar
    #ac, loss = cifar.test(args, model, 0)
    #print ('acc: {}, loss: {}'.format(ac, loss))
    path = 'exp_models/hyper{}_clf_{}_{}.pt'.format(args.dataset, args.exp, acc)
    if args.scratch:
        path = '/scratch/eecs-share/ratzlafn/HyperGAN/' + path
    print ('saving hypernet to {}'.format(path))
    torch.save({'state_dict': model.state_dict()}, path)


def save_hypernet_mnist(args, models, acc):
    netE, W1, W2, W3 = models
    hypernet_dict = {
            'E':  get_net_only(netE),
            'W1': get_net_only(W1),
            'W2': get_net_only(W2),
            'W3': get_net_only(W3),
            }
    path = 'exp_models/hypermnist_{}_{}.pt'.format(args.exp, acc)
    if args.scratch:
        path = '/scratch/eecs-share/ratzlafn/HyperGAN/' + path
    torch.save(hypernet_dict, path)
    print ('Hypernet saved to {}'.format(path))


def save_hypernet_cifar(args, models, acc):
    netE, W1, W2, W3, W4, W5, netD = models
    hypernet_dict = {
            'E':  get_net_only(netE),
            'W1': get_net_only(W1),
            'W2': get_net_only(W2),
            'W3': get_net_only(W3),
            'W4': get_net_only(W4),
            'W5': get_net_only(W5),
            'D': get_net_only(netD),
            }
    path = 'exp_models/hypercifar_{}_{}.pt'.format(args.exp, acc)
    if args.scratch:
        path = '/scratch/eecs-share/ratzlafn/HyperGAN/' + path
    torch.save(hypernet_dict, path)
    print ('Hypernet saved to {}'.format(path))


""" hard coded for mnist experiment dont use generally """
def load_hypernet(path, args=None):
    if args is None:
        args = load_default_args()
    import models.models_mnist_small as hyper
    netE = hyper.Encoder(args).cuda()
    W1 = hyper.GeneratorW1(args).cuda()
    W2 = hyper.GeneratorW2(args).cuda()
    W3 = hyper.GeneratorW3(args).cuda()
    print ('loading hypernet from {}'.format(path))
    d = torch.load(path)
    netE = load_net_only(netE, d['E'])
    W1 = load_net_only(W1, d['W1'])
    W2 = load_net_only(W2, d['W2'])
    W3 = load_net_only(W3, d['W3'])
    return (netE, W1, W2, W3)


def sample_hypernet(hypernet, args=None):
    netE, W1, W2, W3 = hypernet
    x_dist = create_d(300)
    z = sample_d(x_dist, 32)
    codes = netE(z)
    l1 = W1(codes[0])
    l2 = W2(codes[1])
    l3 = W3(codes[2])
    return l1, l2, l3


def weights_to_clf(weights, model, names):
    state = model.state_dict()
    layers = zip(names, weights)
    for i, (name, params) in enumerate(layers):
        name = name + '.weight'
        loader = state[name]
        state[name] = params.detach()
        model.load_state_dict(state)
    return model


def load_default_args():
    parser = argparse.ArgumentParser(description='default hyper-args')
    parser.add_argument('--z', default=128, type=int, help='latent space width')
    parser.add_argument('--ze', default=300, type=int, help='encoder dimension')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--model', default='small2', type=str)
    parser.add_argument('--beta', default=1000, type=int)
    parser.add_argument('--use_x', default=False, type=bool)
    parser.add_argument('--exp', default='0', type=str)
    parser.add_argument('--use_d', default=False, type=str)
    parser.add_argument('--boost', default=10, type=int)

    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='')
    parser.add_argument('--net', type=str, default='small2', metavar='N', help='')
    parser.add_argument('--dataset', type=str, default='mnist', metavar='N', help='')
    parser.add_argument('--mdir', type=str, default='models/', metavar='N', help='')
    parser.add_argument('--scratch', type=bool, default=False, metavar='N', help='')
    parser.add_argument('--ft', type=bool, default=False, metavar='N', help='')
    parser.add_argument('--hyper', type=bool, default=False, metavar='N', help='')
    parser.add_argument('--task', type=str, default='train', metavar='N', help='')

    args = parser.parse_args([])
    return args

