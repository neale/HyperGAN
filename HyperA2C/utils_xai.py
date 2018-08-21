import cv2
import torch
import natsort
import itertools
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
import time
import math
from glob import glob
from scipy.misc import imsave



param_dir = './params/sampled/mnist/test1/'
model_dir = 'encoders/'


def save_model(args, net, optim, m_name):
    path = 'HyperGAN/atari/{}.pt'.format(m_name)
    path = model_dir + path
    torch.save({
        'state_dict': net.state_dict(),
        'optimizer': optim.state_dict(),
        'num_frames': num_frames,
        'mean_reward':reward
        })


def get_net_dict(model, optim):
    net_dict = {
        'state_dict': model.state_dict(),
        'optimizer': optim.state_dict(),
        }
    return net_dict


def open_net_dict(d, model, optim):
    model.load_state_dict(d['state_dict'])
    optim.load_state_dict(d['optimizer'])
    return model, optim


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
