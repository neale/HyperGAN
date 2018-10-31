import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import natsort
import utils
import argparse
from glob import glob

import netdef
import datagen
import models.cifar_clf as models


def load_args():
    parser = argparse.ArgumentParser(description='CIFAR Training Example')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='')
    parser.add_argument('--net', type=str, default='mednet', metavar='N', help='')
    parser.add_argument('--dataset', type=str, default='cifar', help='')
    parser.add_argument('--mdir', type=str, default='models/', help='')
    parser.add_argument('--scratch', type=bool, default=False, help='')
    parser.add_argument('--ft', type=bool, default=False, help='')
    parser.add_argument('--hyper', type=bool, default=False, help='')
    parser.add_argument('--task', type=str, default='train', help='')

    args = parser.parse_args()
    return args


def train(args, model, grad=False):
    if args.dataset == 'cifar':
        train_loader, _ = datagen.load_cifar(args)
    elif args.dataset == 'cifar100':
        train_loader, _ = datagen.load_10_class_cifar100(args)

    train_loss, train_acc = 0., 0.
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    for epoch in range(args.epochs):
        model.train()
        total = 0
        correct = 0
        for i, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            acc = 100. * correct / total
        acc, loss = test(args, model, epoch+1)
    return acc, loss


def test(args, model, epoch=None, grad=False):
    model.eval()
    if args.dataset == 'cifar':
        _, test_loader = datagen.load_cifar(args)
    elif args.dataset == 'cifar100':
        _, test_loader = datagen.load_10_class_cifar100(args)
    test_loss = 0.
    correct = 0.
    criterion = nn.CrossEntropyLoss()
    for i, (data, target) in enumerate(test_loader):
        data, target = data.cuda(), target.cuda()
        output = model(data)
        if grad is False:
            test_loss += criterion(output, target).item()
        else:
            test_loss += criterion(output, target)
        pred = output.data.max(1, keepdim=True)[1]
        output = None
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
    test_loss /= len(test_loader.dataset)
    acc = correct.item() / len(test_loader.dataset)
    if epoch:
        print ('Epoch: {}, Average loss: {}, Accuracy: {}/{} ({}%)'.format(
            epoch, test_loss, correct, len(test_loader.dataset),
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
        save_dir = 'params/cifar/{}/{}/'.format(args.net, l)
        if not os.path.exists(save_dir):
            print ("making ", save_dir)
            os.makedirs(save_dir)
        print ('saving param size: ', params.shape)
        np.save('./params/cifar/{}/{}/{}_{}'.format(args.net, l, l, id), params)


def w_init(model, dist='normal'):
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            if dist == 'normal':
                nn.init.normal_(layer.weight.data)
            if dist == 'uniform':
                nn.init.kaiming_uniform(layer.weight.data)
    return model


def get_network(args):
    if args.net == 'lenet':
        model = models.LeNet().cuda()
    elif args.net == 'mednet':
        model = models.MedNet().cuda()
    else:
        raise NotImplementedError
    return model


def sample_fmodel(args, hypernet, arch):
    w_batch = utils.sample_hypernet_cifar(hypernet)
    rand = np.random.randint(32)
    sample_w = (w_batch[0][rand], w_batch[1][rand], w_batch[2][rand],
            w_batch[3][rand], w_batch[4][rand])
    model = utils.weights_to_clf(sample_w, arch, args.stat['layer_names'])
    return model, sample_w


def measure_models(args, hypernet, weights):
    models = []
    m_conv1, m_conv2, m_conv3, m_linear1, m_linear2 = [], [], [], [], []
    arch = get_network(args)
    with torch.no_grad():
        for i in range(100):
            model, w = sample_fmodel(args, hypernet, arch) 
            m_conv1.append(w[0])
            m_conv2.append(w[1])
            m_conv3.append(w[2])
            m_linear1.append(w[3])
            m_linear2.append(w[4])
        m_conv1 = torch.stack(m_conv1)
        m_conv2 = torch.stack(m_conv2)
        m_conv3 = torch.stack(m_conv3)
        m_linear1 = torch.stack(m_linear1)
        m_linear2 = torch.stack(m_linear2)
        print (m_conv1.shape)
        print (m_conv2.shape)
        print (m_conv3.shape)
        print (m_linear1.shape)
        print (m_linear2.shape)
        
        l2_c1, l2_c2, l2_c3, l2_lin1, l2_lin2 = [], [], [], [], []
        for i in range(100):
            l2_c1.append(np.linalg.norm(m_conv1[i]))
            l2_c2.append(np.linalg.norm(m_conv2[i]))
            l2_c3.append(np.linalg.norm(m_conv3[i]))
            l2_lin1.append(np.linalg.norm(m_linear1[i]))
            l2_lin2.append(np.linalg.norm(m_linear2[i]))
        
        print (np.array(l2_c1).mean(), np.array(l2_c1).std())
        print (np.array(l2_c2).mean(), np.array(l2_c2).std())
        print (np.array(l2_c3).mean(), np.array(l2_c3).std())
        print (np.array(l2_lin1).mean(), np.array(l2_lin1).std())
        print (np.array(l2_lin2).mean(), np.array(l2_lin2).std())
    
    m_conv1, m_conv2, m_conv3, m_linear1, m_linear2 = [], [], [], [], []
    with torch.no_grad():
        for i in range(len(weights)):
            m_conv1.append(weights[i]['conv1.weight'])
            m_conv2.append(weights[i]['conv2.weight'])
            m_conv3.append(weights[i]['conv3.weight'])
            m_linear1.append(weights[i]['fc1.weight'])
            m_linear2.append(weights[i]['fc2.weight'])
        m_conv1 = torch.stack(m_conv1)
        m_conv2 = torch.stack(m_conv2)
        m_conv3 = torch.stack(m_conv3)
        m_linear1 = torch.stack(m_linear1)
        m_linear2 = torch.stack(m_linear2)
        print (m_conv1.shape)
        print (m_conv2.shape)
        print (m_conv3.shape)
        print (m_linear1.shape)
        print (m_linear2.shape)
        
        l2_c1, l2_c2, l2_c3, l2_lin1, l2_lin2 = [], [], [], [], []
        for i in range(len(weights)):
            l2_c1.append(np.linalg.norm(m_conv1[i]))
            l2_c2.append(np.linalg.norm(m_conv2[i]))
            l2_c3.append(np.linalg.norm(m_conv3[i]))
            l2_lin1.append(np.linalg.norm(m_linear1[i]))
            l2_lin2.append(np.linalg.norm(m_linear2[i]))

        print (np.array(l2_c1).mean(), np.array(l2_c1).std())
        print (np.array(l2_c2).mean(), np.array(l2_c2).std())
        print (np.array(l2_c3).mean(), np.array(l2_c3).std())
        print (np.array(l2_lin1).mean(), np.array(l2_lin1).std())
        print (np.array(l2_lin2).mean(), np.array(l2_lin2).std())
    return 


def run_measure(args, path):
    hypernet = utils.load_hypernet_cifar(path)
    mpaths = glob('./exp_models/cifar/mednet/*.pt')
    models = []
    print (mpaths)
    for path in mpaths:
        models.append(torch.load(path)['state_dict'])
    measure_models(args, hypernet, models)


def run_model_search(args, path):
    for i in range(0, 2):
        print ("\nRunning CIFAR Model {}...".format(i))
        model = get_network(args)
        model = w_init(model, 'normal')
        print (model)
        acc, loss = train(args, model)
        extract_weights_all(args, model, i)
        torch.save({'state_dict': model.state_dict()},
                path+'cifar_clf_{}.pt'.format(acc))


""" Load a batch of networks to extract weights """
def load_models(args, path):
    model = get_network(args)
    paths = glob(path + '*.pt'.format(args.net))
    paths = natsort.natsorted(paths)
    for i, path in enumerate(paths):
        print ("loading model {}".format(path))
        ckpt = torch.load(path)
        state = ckpt['state_dict']
        try: 
            model.load_state_dict(state)
        except RuntimeError:
            model_dict = model.state_dict()
            filtered = {k: v for k, v in state.items() if k in model_dict}
            model_dict.update(filtered)
            model.load_state_dict(filtered)

        acc, loss = test(args, model, 0)
        print ('Test0 Acc: {}, Loss: {}'.format(acc, loss))
        acc, loss = train(args, model)
        print ('Train Acc: {}, Loss: {}'.format(acc, loss))
        #extract_weights_all(args, model, i)


if __name__ == '__main__':
    args = load_args()
    args.stat = netdef.nets()[args.net]
    args.shapes = args.stat['shapes']
    if args.scratch:
        path = './'
        if args.hyper:
            path = path + './exp_models'
    else:
        path = './'
    if args.task == 'test':
        load_models(args, path)
    elif args.task == 'train':
        run_model_search(args, path)
    elif args.task == 'measure':
        run_measure(args, path)
    else:
        raise NotImplementedError

