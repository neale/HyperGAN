import os
import numpy as np
from glob import glob

def param_generator(args, data, batch_size):
    if args.dataset == 'mnist':
        if args.size == '1x':
            if args.layer == 'conv1':
                filter_shape = (32, 3, 3)
            if args.layer == 'conv2':
                filter_shape = (64, 3, 3)
        elif args.size == 'wide':
            if args.layer == 'conv1':
                filter_shape = (128, 3, 3)
            if args.layer == 'conv2':
                filter_shape = (256, 3, 3)
        elif args.size == 'wide7':
            if args.layer == 'conv1':
                filter_shape = (128, 7, 7)
            if args.layer == 'conv2':
                filter_shape = (256, 7, 7)
    elif args.dataset == 'cifar':
        if args.size in ['presnet', 'resnet']:
            filter_shape = (512, 3, 3)
        if args.size == '1x':
            filter_shape = (128, 3, 3)

    filter_volume = data
    filters = filter_volume.reshape((-1, *filter_shape))
    rng_state = np.random.get_state()
    np.random.set_state(rng_state)
    print (filters.shape)
    filters = filters[len(filters)%batch_size:]
    def get_epoch():
        np.random.shuffle(filters)
        params_batches = filters.reshape(-1, batch_size, *filter_shape)

        for i in range(len(params_batches)):
            yield (np.copy(params_batches[i]))
    return get_epoch


def load(args, layer='conv2'):
    if args.dataset == 'mnist':
        if args.size == '1x':
            if args.layer == 'conv1':
                pdir = './params/mnist/1x/conv1/'
                pshape = (32, 1, 3, 3)
            if args.layer == 'conv2':
                pdir = './params/mnist/1x/conv2/'
                pshape = (64, 32, 3, 3)
        elif args.size == 'wide':
            if args.layer == 'conv1':
                pdir = './params/mnist/wide/conv1/'
                pshape = (128, 1, 3, 3)
            if args.layer == 'conv2':
                pdir = './params/mnist/wide/conv2/'
                pshape = (256, 128, 3, 3)
        elif args.size == 'wide7':
            if args.layer == 'conv1':
                pdir = './params/mnist/wide7/conv1/'
                pshape = (128, 1, 7, 7)
            if args.layer == 'conv2':
                pdir = './params/mnist/wide7/conv2/'
                pshape = (256, 128, 7, 7)

    elif args.dataset == 'cifar':
        if args.size == '1x':
            pdir = './params/cifar/1x/conv2/'
            pshape = (128, 64, 3, 3)
        if args.size in ['presnet', 'resnet']:
            pdir = './params/cifar/{}/layer4.1.conv2/'.format(args.size)
            pshape = (512, 512, 3, 3)

    paths = glob(pdir+'*.npy')
    print (len(paths))
    data = np.zeros((len(paths), *pshape))
    for i in range(len(paths)):
        data[i] = np.load(paths[i])
    print (data.shape)
    
    len_t = int(len(data) * .9)
    train_data = data[:len_t]
    val_data = data[len_t:]
    
    return (param_generator(args, train_data, args.batch_size), 
            param_generator(args, val_data, args.batch_size))
