import os
import sys
import argparse
import natsort
import numpy as np
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utils
import models.mnist_clf as models
import models.models_mnist_small as hyper
import datagen
import netdef
from torchvision.utils import save_image

def load_args():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='')
    parser.add_argument('--net', type=str, default='small2', metavar='N', help='')
    parser.add_argument('--dataset', type=str, default='mnist', metavar='N', help='')
    parser.add_argument('--mdir', type=str, default='models/', metavar='N', help='')
    parser.add_argument('--scratch', type=bool, default=False, metavar='N', help='')
    parser.add_argument('--ft', type=bool, default=False, metavar='N', help='')
    parser.add_argument('--hyper', type=bool, default=False, metavar='N', help='')
    parser.add_argument('--task', type=str, default='train', metavar='N', help='')

    args = parser.parse_args()
    return args


def train(args, model, grad=False):
    if args.dataset =='mnist':
        train_loader, _ = datagen.load_mnist(args)
    elif args.dataset == 'fashion_mnist':
        train_loader, _ = datagen.load_fashion_mnist(args)
    train_loss, train_acc = 0., 0.
    criterion = nn.CrossEntropyLoss()
    if args.ft:
        for child in list(model.children())[:2]:
            print('removing {}'.format(child))
            for param in child.parameters():
                param.requires_grad = False
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    for epoch in range(args.epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        acc, loss = test(args, model, epoch)
    return acc, loss


def test(args, model, epoch=None, grad=False):
    model.eval()
    if args.dataset =='mnist':
        _, test_loader = datagen.load_mnist(args)
    elif args.dataset == 'fashion_mnist':
        _, test_loader = datagen.load_fashion_mnist(args)
    test_loss = 0
    correct = 0.
    criterion = nn.CrossEntropyLoss()
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        output = model(data)
        if grad is False:
            test_loss += criterion(output, target).item() # sum up batch loss
        else:
            test_loss += criterion(output, target)

        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    acc = (correct.float() / len(test_loader.dataset)).item()
    print (acc)

    if epoch:
        print('Average loss: {}, Accuracy: {}/{} ({}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return acc, test_loss


def extract_weights_all(args, model, id):
    state = model.state_dict()
    conv_names = [x for x in list(state.keys()) if 'conv' in x]
    fc_names = [x for x in list(state.keys()) if 'linear' in x]
    cnames = [x for x in conv_names if 'weight' in x] 
    fnames = [x for x in fc_names if 'weight' in x]
    names = cnames + fnames
    print (names)
    for i, name in enumerate(names):
        print (name)
        l = name[:-7]
        conv = state[name]
        params = conv.cpu().numpy()
        save_dir = 'params/mnist/{}/{}/'.format(args.net, l)
        if not os.path.exists(save_dir):
            print ("making ", save_dir)
            os.makedirs(save_dir)
        print ('saving param size: ', params.shape)
        np.save('./params/mnist/{}/{}/{}_{}'.format(args.net, l, l, id), params)


def sample_fmodel(args, hypernet, arch):
    w_batch = utils.sample_hypernet(hypernet)
    rand = np.random.randint(32)
    sample_w = (w_batch[0][rand], w_batch[1][rand], w_batch[2][rand])
    model = utils.weights_to_clf(sample_w, arch, args.stat['layer_names'])
    return model, sample_w


def measure_models(args, hypernet, weights):
    
    models = []
    m_conv1, m_conv2, m_linear = [], [], []
    arch = get_network(args)
    if args.hyper:
        with torch.no_grad():
            for i in range(100):
                model, weights = sample_fmodel(args, hypernet, arch) 
                m_conv1.append(weights[0])
                m_conv2.append(weights[1])
                m_linear.append(weights[2])
            m_conv1 = torch.stack(m_conv1)
            m_conv2 = torch.stack(m_conv2)
            m_linear = torch.stack(m_linear)
            print (m_conv1.shape)
            print (m_conv2.shape)
            print (m_linear.shape)
            
            l2_c1, l2_c2, l2_lin = [], [], []
            for i in range(100):
                l2_c1.append(np.linalg.norm(m_conv1[i]))
                l2_c2.append(np.linalg.norm(m_conv2[i]))
                l2_lin.append(np.linalg.norm(m_linear[i]))
            print (l2_c1)
            print (l2_c2)
            print (l2_lin)
    else:
        with torch.no_grad():
            for i in range(len(weights)):
                m_conv1.append(weights[i]['conv1.0.weight'])
                m_conv2.append(weights[i]['conv2.0.weight'])
                m_linear.append(weights[i]['linear.weight'])
            m_conv1 = torch.stack(m_conv1)
            m_conv2 = torch.stack(m_conv2)
            m_linear = torch.stack(m_linear)
            print (m_conv1.shape)
            print (m_conv2.shape)
            print (m_linear.shape)
            
            l2_c1, l2_c2, l2_lin = [], [], []
            for i in range(len(weights)):
                l2_c1.append(np.linalg.norm(m_conv1[i]))
                l2_c2.append(np.linalg.norm(m_conv2[i]))
                l2_lin.append(np.linalg.norm(m_linear[i]))
            print (l2_c1)
            print (l2_c2)
            print (l2_lin)
    return 
    print (m1_conv.shape, m1_linear.mean(0).shape)
    
    l1_conv = (m1_conv - m2_conv).abs().sum()
    l1_linear = (m1_linear - m2_linear).abs().sum()
    print ("\nL1 Dist: {} - {}".format(l1_conv, l1_linear))

    l2_conv = np.linalg.norm(m1_conv - m2_conv)
    l2_linear = np.linalg.norm(m1_linear - m2_linear)
    print ("L2 Dist: {} - {}".format(l2_conv, l2_linear))

    linf_conv = np.max((m1_conv-m2_conv).abs().cpu().numpy())
    linf_linear = np.max((m1_linear-m2_linear).abs().cpu().numpy())
    print ("Linf Dist: {} - {}".format(linf_conv, linf_linear))

    cov_m1_m2_conv = np.cov(m1_conv, m2_conv)[0, 1]
    cov_m1_m2_linear = np.cov(m1_linear, m2_linear)[0, 1]
    print ("Cov m1-m2: {} - {}".format(cov_m1_m2_conv, cov_m1_m2_linear))

    cov_m1_conv_linear = np.cov(m1_conv, m1_linear.mean(0))[0, 1]
    cov_m2_conv_linear = np.cov(m2_conv, m2_linear.mean(0))[0, 1]
    print ("Cov m1-conv-linear: {} - {}\n\n".format(cov_m1_conv_linear, cov_m2_conv_linear))
   
    return
    utils.plot_histogram([m1_conv.view(-1).cpu().numpy(), m2_conv.view(-1).cpu().numpy()],
            save=False)
    utils.plot_histogram([m1_linear.view(-1).cpu().numpy(), m2_linear.view(-1).cpu().numpy()],
            save=False)
    
    for l, name in [(m1_conv, 'conv1.0'), (m1_linear, 'linear')]:
        params = l.cpu().numpy()
        save_dir = 'params/mnist/{}/{}'.format(args.net, name)
        if not os.path.exists(save_dir):
            print ("making ", save_dir)
            os.makedirs(save_dir)
        path = '{}/{}_{}.npy'.format(save_dir, name, i)
        print (i)
        print ('saving param size: ', params.shape, 'to ', path)
        np.save(path, params)


""" 
init all weights in the net from a normal distribution
Does not work for ResNets 
"""
def w_init(model, dist='normal'):
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            if dist == 'normal':
                nn.init.normal(layer.weight.data)
            if dist == 'uniform':
                nn.init.kaiming_uniform(layer.weight.data)
    return model


""" returns instance of specific model without weights """
def get_network(args):
    if args.net == 'small':
        model = models.Small().cuda()
    elif args.net == 'small2':
        model = models.Small2().cuda()
    elif args.net == 'mc':
        model = models.Small2_MC().cuda()
    else:
        raise NotImplementedError
    return model


def run_measure(args, path):
    paths = glob(path + '/*.pt')
    path = './hypermnist_0_0.984871875.pt'
    #path = './hypermnist_disc_0.97311875.pt'
    #path = './hypermnist_disc_0.97581875.pt'
    path = './hypermnist_disc_0.8857125.pt'
    hypernet = utils.load_hypernet(path)
    mpaths = ['mnist_model_small2_0.pt',
             'mnist_model_small2_1.pt',
             'mnist_model_small2_2.pt',
             'mnist_model_small2_3.pt',
             'mnist_model_small2_4.pt',
             #'mnist_model_small2_5.pt',
             #'mnist_model_small2_6.pt',
             #'mnist_model_small2_7.pt',
             #'mnist_model_small2_8.pt',
             #'mnist_model_small2_9.pt'
             ]
    models = []
    for path in mpaths:
        model = get_network(args)
        models.append(torch.load(path))
    measure_models(args, hypernet, models)


""" train and save models and their weights """
def run_model_search(args, path):

    for i in range(0, 5):
        print ("\nRunning MNIST Model {}...".format(i))
        model = get_network(args)
        print (model)
        model = w_init(model, 'normal')
        acc, loss = train(args, model)
        #extract_weights_all(args, model, i)
        torch.save(model.state_dict(), './mc_model_{}_{}.pt'.format(args.net, i, acc))


""" Load a batch of networks to extract weights """
def load_models(args, path):
   
    model = get_network(args)
    paths = glob(path + '*.pt')
    print (path)
    paths = [path for path in paths if 'mnist' in path]
    natpaths = natsort.natsorted(paths)
    accs = []
    losses = []
    natpaths = [x for x in natpaths if 'hypermnist_mi_0.987465625' in x]
    for i, path in enumerate(natpaths):
        print ("loading model {}".format(path))
        if args.hyper:
            hn = utils.load_hypernet(path)
            for i in range(10):
                samples = utils.sample_hypernet(hn)
                print ('sampled a batches of {} networks'.format(len(samples[0])))
                for i, sample in enumerate(zip(samples[0], samples[1], samples[2])):
                    model = utils.weights_to_clf(sample, model, args.stat['layer_names'])
                    acc, loss = test(args, model)
                    print (i, ': Test Acc: {}, Loss: {}'.format(acc, loss))
                    accs.append(acc)
                    losses.append(loss)
                    #acc, loss = train(args, model)
                    #print ('Test1 Acc: {}, Loss: {}'.format(acc, loss))
                    #extract_weights_all(args, model, i)
            print(accs, losses)
        else:
            ckpt = torch.load(path)
            state = ckpt['state_dict']
            try:
                model.load_state_dict()
            except RuntimeError:
                model_dict = model.state_dict()
                filtered = {k:v for k, v in state.items() if k in model_dict}
                model_dict.update(filtered)
                model.load_state_dict(filtered)



if __name__ == '__main__':
    args = load_args()
    args.stat = netdef.nets()[args.net]
    args.shapes = netdef.nets()[args.net]['shapes']
    if args.scratch:
        path = '/scratch/eecs-share/ratzlafn/HyperGAN/'
        if args.hyper:
            path = path +'exp_models'
    else:
        path = './'

    if args.task == 'test':
        load_models(args, path)
    elif args.task =='train':
        run_model_search(args, path)
    elif args.task == 'measure':
        run_measure(args, path)

    else:
        raise NotImplementedError
