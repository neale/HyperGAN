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

import torch.nn as nn
import torch.nn.init as init


param_dir = './params/sampled/mnist/test1/'


def save_model(net, optim, epoch, path):
    state_dict = net.state_dict()
    torch.save({
        'epoch': epoch + 1,
        'state_dict': state_dict,
        'optimizer': optim.state_dict(),
        }, path)


def load_model(net, optim, path):
    ckpt = torch.load(path)
    epoch = ckpt['epoch']
    net.load_state_dict(ckpt['state_dict'])
    optim.load_state_dict(ckpt['optimizer'])
    return net, optim, epoch


def plot_histogram(x, save=False, id=None):
    x = x.flatten()
    plt.ion()
    n, bins, patches = plt.hist(x, 50, density=True, alpha=0.75)
    plt.xlabel('params')
    plt.ylabel('probability')
    plt.title('Distribution of conv2 parameters')
    plt.grid(True)
    if save is True:
        plt.draw()
        plt.pause(1.1)
    else:
        plt.show()


def dataset_iterator(args):
    train_gen, dev_gen = datagen.load(args)
    return (train_gen, dev_gen)


def inf_train_gen(train_gen):
    while True:
        for params in train_gen():
            yield params


def load_params(flat=True):
    paths = glob(param_dir+'/*.npy')
    paths = natsort.natsorted(paths)
    s = np.load(paths[0]).shape
    print (s)
    params = np.zeros((len(paths), *s))
    print (params.shape)
    for i in range(len(paths)):
        params[i] = np.load(paths[i])

    if flat is True:
        res = params.flatten()
        params = res
    return res


def generate_samples(iter, G, path, args):
    batch_size = args.batch_size
    if args.dataset == 'mnist':
        if args.size == '1x':
            if args.layer == 'conv1':
                shape = (32, 1, 3, 3)
            if args.layer == 'conv2':
                shape = (64, 32, 3, 3)
        if args.size == 'wide':
            if args.layer == 'conv1':
                shape = (128, 1, 3, 3)
            if args.layer == 'conv2':
                shape = (256, 128, 3, 3)
        if args.size == 'wide7':
            if args.layer == 'conv1':
                shape = (128, 1, 7, 7)
            if args.layer == 'conv2':
                shape = (256, 128, 7, 7)
    elif args.dataset == 'cifar':
        if args.size in ['presnet', 'resnet']:
            shape = (512, 512, 3, 3)
        if args.size == '1x':
            shape = (128, 64, 3, 3)
    params = np.zeros(shape)
    fixed_noise = torch.randn(shape[1], args.dim).cuda()
    noisev = autograd.Variable(fixed_noise, volatile=True)
    if args.layer == 'conv1':
        samples = G(noisev)[0].view(*shape)
    else:
        samples = G(noisev).view(*shape)
    params = samples.cpu().data.numpy()
    np.save(path+'/params_iter_{}.npy'.format(iter), params)
    acc = test_samples(args, iter, params)
    return acc


def test_samples(args, iter, params):
    # take random model
    id = np.random.randint(100)
    paths = natsort.natsorted(glob('models/{}/{}/*.pt'.format(
        args.dataset, args.size)))
    if args.dataset == 'mnist':
        model = mnist.WideNet().cuda()
        test = mnist.test
        layer_name = args.layer+'.0.weight'
    elif args.dataset == 'cifar':
        if args.size == 'presnet':
            model = PreActResNet18().cuda()
        if args.size == 'resnet':
            model = ResNet18().cuda()
        test = cifar.test
        if args.size in ['presnet', 'resnet']:
            layer_name = 'layer4.1.conv2'
        layer_name = args.layer+'.weight'
    print (paths[id])
    model.load_state_dict(torch.load(paths[id]))
    state = model.state_dict()
    conv2 = state[layer_name]
    state[layer_name] = torch.Tensor(params).cuda()
    plot_histogram(params, save=True, id=str(id)+'-'+str(iter))
    model.load_state_dict(state)
    acc, loss = test(model)
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


if __name__ =='__main__':

    params = load_params()
    plot_histogram(params, True)
