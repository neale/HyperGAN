import os
import numpy as np
from glob import glob
import utils
import torch
import torchvision
from torchvision import datasets, transforms


def load_fashion_mnist():
    path = 'data_f'
    if args.scratch:
        path = '/scratch/eecs-share/ratzlafn/' + path
    torch.cuda.manual_seed(1)
    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(path, train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ])),
                batch_size=64, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(path, train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ])),
                batch_size=64, shuffle=True, **kwargs)
    return train_loader, test_loader


def load_cifar():
    path = './data_c'
    if args.scratch:
        path = '/scratch/eecs-share/ratzlafn/' + path
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])  
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])  
    trainset = torchvision.datasets.CIFAR10(root=path, train=True,
            download=True,
            transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
            shuffle=True,
            num_workers=2)
    testset = torchvision.datasets.CIFAR10(root=path, train=False,
            download=True,
            transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
            shuffle=False, num_workers=2)
    return trainloader, testloader


def load_mnist(args):
    torch.cuda.manual_seed(1)
    kwargs = {'num_workers': 1, 'pin_memory': True}
    path = 'data_m/'
    if args.scratch:
        path = '/scratch/eecs-share/ratzlafn/' + path
    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ])),
                batch_size=32, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ])),
            batch_size=32, shuffle=True, **kwargs)
    return train_loader, test_loader


def param_generator(args, data):

    filter_volume = data
    layer_size = data.shape[2]
    filter_shape = args.shapes[args.id]
    filters = filter_volume.reshape((-1, *filter_shape))
    # print ('filters a', filters.shape)
    rng_state = np.random.get_state()
    np.random.set_state(rng_state)
    # filters = filters[len(filters)%layer_size:]
    #print ('layer size ', layer_size)
    #print ('filter shape', filter_shape)
    #print ('filters', filters.shape)
    def get_epoch():
        # np.random.shuffle(filters)
        params_batches = filters.reshape(-1, *filter_shape)
        # print (params_batches.shape)
        for i in range(len(params_batches)):
            yield (np.copy(params_batches[i]))
    return get_epoch


def load(args):
    generators = []
    for i in range(args.stat['n_layers']):
        args.id = id = i
        pdir = './params/{}/{}/{}/'.format(
                args.dataset, args.model, args.stat['layer_names'][id])
        pshape = args.shapes[id]
        paths = glob(pdir+'*.npy')
        data = np.zeros((len(paths), *pshape))
        for i in range(len(paths)):
            data[i] = np.load(paths[i])
        # len_t = int(len(data) * .9)
        train_data = data  # [:len_t]
        # val_data = data[len_t:]
        generators.append(param_generator(args, train_data))
    return generators

    # return (param_generator(args, train_data, id), 
    #         param_generator(args, val_data, id))
