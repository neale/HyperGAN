import torch
import natsort
import datagen
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import torch.autograd as autograd
import itertools
import cv2
import numpy as np

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

import torch.nn as nn
import torch.nn.init as init


param_dir = './params/sampled/mnist/test1/'
model_dir = '/data0/models/HyperGAN/models/'


def save_model(args, net, optim, ex_name):
    path = 'HyperGAN/{}/{}/{}/{}'.format(args.dataset, ex_name, args.model, net.name)
    path = model_dir + path
    state_dict = net.state_dict()
    torch.save({
        'state_dict': state_dict,
        'optimizer': optim.state_dict(),
        }, path)


def load_model(args, net, optim, ex_name):
    path = 'HyperGAN/{}/{}/{}/{}'.format(args.dataset, ex_name, args.model, net.name)
    path = model_dir + path
    ckpt = torch.load(path)
    net.load_state_dict(ckpt['state_dict'])
    optim.load_state_dict(ckpt['optimizer'])
    return net, optim


def plot_histogram(x, save=False, id=None):
    if save is True:
        plt.ion()
    if type(x) is list:
        for i in range(len(x)):
            n, bins, patches = plt.hist(x[i], 50, density=True, alpha=0.75, label=str(i))
    else:
        n, bins, patches = plt.hist(x, 50, density=True, alpha=0.75)

    plt.xlabel('params')
    plt.legend(loc='upper right')
    plt.ylabel('probability')
    plt.title(id)
    plt.grid(True)
    if save is True:
        plt.draw()
        plt.pause(1.1)
    else:
        plt.show()


def dataset_iterator(args, id):
    train_gen, dev_gen = datagen.load(args, id)
    return (train_gen, dev_gen)


def inf_train_gen(train_gen):
    if type(train_gen) is list:
        while True:
            for (p1) in (train_gen[0](), train_gen[1]()):
                yield (p1)
    else:
        while True:
            for params in train_gen():
                yield params


def load_params(flat=True):
    paths = glob(param_dir+'/*.npy')
    paths = natsort.natsorted(paths)
    s = np.load(paths[0]).shape
    # print (s)
    params = np.zeros((len(paths), *s))
    # print (params.shape)
    for i in range(len(paths)):
        params[i] = np.load(paths[i])

    if flat is True:
        res = params.flatten()
        params = res
    return res


def save_samples(args, samples, iter, path):
    # lets view the first filter
    filters = samples[:, 0, :, :]
    filters = filters.unsqueeze(3)
    grid_img = grid(16, 8, filters, margin=2)
    im_path = 'plots/{}/{}/filters/{}.png'.format(args.dataset, args.model, iter)
    cv2.imwrite(im_path, grid_img)
    return


def test_samples(args, params, train=False):
    # take random model
    paths = natsort.natsorted(glob(model_dir+'{}/{}/*.pt'.format(
        args.dataset, args.model)))
    id = np.random.randint(len(paths)-1)
    if args.dataset == 'cifar':
        model_fn = getattr(cifar, args.stat['name'])
    else:
        model_fn = getattr(mnist, args.stat['name'])
    model = model_fn().cuda()
    model.load_state_dict(torch.load(paths[id]))
    if train is True:
        if args.dataset == 'cifar':
            fn = cifar.train
        else:
            fn = mnist.train
    else:
        if args.dataset == 'cifar':
            fn = cifar.test
        else:
            fn = mnist.test
    #oracle_acc, oracle_loss = fn(model, grad=False)
    state = model.state_dict()
    if args.layer == 'all':
        assert type(params) == list
        layers = zip(args.stat['layer_names'], params)
        for i, (name, params) in enumerate(layers):
            name = name + '.weight'
            loader = state[name]
            state[name] = params.data
            assert state[name].equal(loader) == False
            # plot_histogram(params, save=True, id=str(id)+'-'+str(iter))
            model.load_state_dict(state)
        acc, loss = fn(model, grad=True)
    else:
        raise NotImplementedError

    return acc, loss


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def grid(w, h, imgs, margin):
    n = w*h
    img_h, img_w, img_c = imgs[0].shape
    m_x = 0
    m_y = 0
    if margin is not None:
        m_x = int(margin)
        m_y = m_x
    imgmatrix = np.zeros((img_h * h + m_y * (h - 1),
        img_w * w + m_x * (w - 1),
        img_c),
        np.uint8)
    imgmatrix.fill(255)    

    positions = itertools.product(range(w), range(h))
    for (x_i, y_i), img in zip(positions, imgs):
        x = x_i * (img_w + m_x)
        y = y_i * (img_h + m_y)
        imgmatrix[y:y+img_h, x:x+img_w, :] = img
    return imgmatrix


if __name__ =='__main__':

    params = load_params()
    plot_histogram(params, True)
