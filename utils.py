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

import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init


param_dir = './params/sampled/mnist/test1/'

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
                datashape = (3, 3, 128)
            if args.layer == 'conv2':
                datashape = (3, 3, 256)
        if args.size == 'wide7':
            if args.layer == 'conv1':
                datashape = (7, 7, 128)
            if args.layer == 'conv2':
                datashape = (7, 7, 256)
    elif args.dataset == 'cifar':
        datashape = (3, 3, 128)

    alpha = torch.rand(batch_size, 1)
    # if args.layer == 'conv1':
    #     alpha = alpha.expand(real_data.size()).cuda()
    alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size))
    alpha = alpha.contiguous().view(batch_size, *(datashape[::-1])).cuda()
    interpolates = alpha * real_data + ((1 - alpha) * gen_data)
    interpolates = interpolates.cuda()
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
        shape = (128, 64, 3, 3)
    params = np.zeros(shape)
    fixed_noise = torch.randn(batch_size, args.dim).cuda()
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
    id = np.random.randint(200)
    paths = natsort.natsorted(glob('models/{}/{}/*.pt'.format(
        args.dataset, args.size)))
    if args.dataset == 'mnist':
        model = mnist.WideNet7().cuda()
        test = mnist.test
        layer_name = args.layer+'.0.weight'
    elif args.dataset == 'cifar':
        model = cifar.WideNet().cuda()
        test = cifar.test
        layer_name = args.layer+'.weight'
    model.load_state_dict(torch.load(paths[id]))
    state = model.state_dict()
    conv2 = state[layer_name]
    state[layer_name] = torch.Tensor(params).cuda()
    plot_histogram(params, save=True, id=str(id)+'-'+str(iter))
    model.load_state_dict(state)
    acc = test(model, 0)
    return acc



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


