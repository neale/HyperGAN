import os
import numpy as np
from glob import glob
import utils
import torch
import torchvision
import pickle
from torchvision import datasets, transforms


def load_mnist(args):
    torch.cuda.manual_seed(1)
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': True}
    path = 'data_m/'
    if args.scratch:
        path = '/scratch/eecs-share/ratzlafn/' + path
    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ])),
                #batch_size=32, shuffle=True, **kwargs)
                batch_size=32, **kwargs)
    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ])),
            batch_size=32, shuffle=True, **kwargs)
    return train_loader, test_loader


def load_fashion_mnist():
    path = 'data_f'
    if args.scratch:
        path = '/scratch/eecs-share/ratzlafn/' + path
    torch.cuda.manual_seed(1)
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': True}
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


def load_cifar(args):
    path = './data_c'
    if args.scratch:
        path = '/scratch/eecs-share/ratzlafn/' + path
    kwargs = {'num_workers': 2, 'pin_memory': True, 'drop_last': True}
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
            download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
            shuffle=True, **kwargs)
    testset = torchvision.datasets.CIFAR10(root=path, train=False,
            download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
            shuffle=False, **kwargs)
    return trainloader, testloader


def load_cifar100(args):
    path = './data_c100'
    if args.scratch:
        path = '/scratch/eecs-share/ratzlafn/' + path
    kwargs = {'num_workers': 2, 'pin_memory': True, 'drop_last': True}
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
    trainset = torchvision.datasets.CIFAR100(root=path, train=True,
            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
            shuffle=True, **kwargs)
    testset = torchvision.datasets.CIFAR100(root=path, train=False,
            download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
            shuffle=False, **kwargs)
    return trainloader, testloader

"""
def load_10_class_cifar100(args):
    # arbitrarily choose classes 
    d_train = '/scratch/eecs-share/ratzlafn/cifar100-python/train'
    with open(d_train, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    data = zip(d[b'fine_labels'], d[b'data'], d[b'filenames'])
    all_data = []
    for (label, x, fn) in enumerate(data):
        all_data.append(
"""
def load_10_class_cifar100(args):
    path = 'cifar-100-python'
    if args.scratch:
        path = '/scratch/eecs-share/ratzlafn/' + path
    kwargs = {'num_workers': 2, 'pin_memory': True, 'drop_last': True}
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
    trainset = torchvision.datasets.ImageFolder(root=path+'/train',
            transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
            shuffle=True, **kwargs)
    testset = torchvision.datasets.ImageFolder(root=path+'/test',
            transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
            shuffle=False, **kwargs)
    return trainloader, testloader


