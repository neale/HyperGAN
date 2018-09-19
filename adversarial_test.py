import matplotlib
matplotlib.use('Agg')
import os
import sys
import argparse
import natsort
import numpy as np
from glob import glob
from math import e, log
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import statsmodels.api as sm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image

import utils
import netdef
import datagen
import attacks
import models.mnist_clf as models
import models.models_mnist_small as hyper

import logging
import foolbox
from foolbox.criteria import Misclassification
foolbox_logger = logging.getLogger('foolbox')
logging.disable(logging.CRITICAL);
import warnings
warnings.filterwarnings("ignore")

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


def plot_entropy(args, a, eps):
    a = np.concatenate(a).ravel()
    print (a)
    np.save('/scratch/eecs-share/ratzlafn/{}_ent_FGSM_{}.npy'.format(args.dataset, eps), a)
    n, bins, patches = plt.hist(a, 50, normed=1, facecolor='green', alpha=0.75)
    mu = np.mean(a)
    sigma = np.std(a)
    y = mlab.normpdf(bins, mu, sigma)
    l = plt.plot(bins, y, 'r--', linewidth=1)
    plt.title('predictive entropy with eps: '.format(eps))
    print ('ecdf with {} samples'.format(len(a)))
    plt.savefig('/scratch/eecs-share/ratzlafn/{}_fgsm_ent_{}.png'.format(args.dataset, eps))


def entropy(y, base=None):
    if len(y) <= 1:
        return 0
    value, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    n_classes = np.count_nonzero(probs)
    if n_classes <= 1:
        return 0
    entropy = 0.
    base = e if base is None else base
    for i in probs:
        entropy -= i * log(i, base)
    return entropy

# Basically I need to modify an attack so that it takes the
# gradient under a new network for each iteration: IFGSM
# One step attacks, do they transfer, no need to modify attacks here. 
def sample_fmodel(hypernet, arch):
    w_batch = utils.sample_hypernet(hypernet)
    rand = np.random.randint(32)
    sample_w = (w_batch[0][rand], w_batch[1][rand], w_batch[2][rand])
    model = utils.weights_to_clf(sample_w, arch, args.stat['layer_names'])
    model.eval()
    fmodel = attacks.load_model(model)
    return model, fmodel


def unnormalize(x):
    x *= .3081
    x += .1307
    return x


def normalize(x):
    x -= .1307
    x /= .3081
    return x


def sample_adv_batch(data, target, fmodel, eps, attack):
    missed = 0
    inter, adv, y = [], [], []

    for i in range(32):
        input = unnormalize(data[i].cpu().numpy())
        x_adv = attack(input, target[i].item(),
                #binary_search=False,
                #stepsize=2,
                #epsilon=eps) #normalized
                epsilons=[eps]) #normalized
        px = np.argmax(fmodel.predictions(normalize(input))) #renormalized input
        # Failure conditions
        if (x_adv is None) or (px != target[i].item()):
            missed += 1
            continue
        inter.append(np.argmax(fmodel.predictions(x_adv)))
        assert (inter[-1] != px and inter[-1] != target[i].item())
        adv.append(torch.from_numpy(x_adv))
        y.append(target[i])
    if adv == []:
        adv_batch, target_batch = None, None
    else:
        adv_batch = torch.stack(adv).cuda()
        target_batch = torch.stack(y).cuda()
    return adv_batch, target_batch, inter


# we want to estimate performance of a sampled model on adversarials
def run_adv_hyper(args, hypernet):
    arch = get_network(args)
    model_base, fmodel_base = sample_fmodel(hypernet, arch)
    criterion = Misclassification()
    fgs = foolbox.attacks.FGSM(fmodel_base, criterion)
    _, test_loader = datagen.load_mnist(args)
    adv, y = [],  []
    for eps in [0.1]:
        total_adv = 0
        acc, _accs = [], []
        _vars, _stds, _ents = [], [], []
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            adv_batch, target_batch, _ = sample_adv_batch(
                    data, target, fmodel_base, eps, fgs)
            if adv_batch is None:
                continue
            output = model_base(adv_batch)
            pred = output.data.max(1, keepdim=True)[1]
            correct = pred.eq(target_batch.data.view_as(pred)).long().cpu().sum()
            n_adv = len(target_batch) - correct.item()
            total_adv += n_adv
            padv = np.argmax(fmodel_base.predictions(
                adv_batch[0].cpu().numpy()))

            sample_adv, pred_labels, logits = [], [], []
            for _ in range(100):
                model, fmodel = sample_fmodel(hypernet, arch) 
                output = model(adv_batch)
                pred = output.data.max(1, keepdim=True)[1]
                correct = pred.eq(target_batch.data.view_as(pred)).long().cpu().sum()
                acc.append(correct.item())
                n_adv_sample = len(target_batch)-correct.item()
                sample_adv.append(n_adv_sample)
                pred_labels.append(pred.view(pred.numel()))
                logits.append(F.softmax(output, dim=1))
            p_labels = torch.stack(pred_labels).float().transpose(0, 1)
            logits = torch.stack(logits)
            acc = torch.tensor(acc, dtype=torch.float)
            _accs.append(torch.mean(acc))
            _vars.append(p_labels.var(1).mean())
            _stds.append(p_labels.std(1).mean())
            #print (logits[0])
            #ent = entropy(logits.detach())
            #print (ent.shape)
            #_ents.append(ent)
            _ents.append(np.apply_along_axis(entropy, 1, p_labels.detach()))
            acc, adv, y = [], [], []

        plot_entropy(args, _ents, eps)
        print ('Eps: {}, Adv: {}/{}, var: {}, std: {}'.format(eps,
            total_adv, len(test_loader.dataset), torch.tensor(_vars).mean(),
            torch.tensor(_stds).mean()))


def run_adv_model(args, model):
    model.eval()
    fmodel = attacks.load_model(model)
    criterion = Misclassification()
    fgs = foolbox.attacks.BIM(fmodel)
    _, test_loader = datagen.load_mnist(args)
    adv, y, inter = [],  [], []
    acc, accs = [], []
    total_adv, total_correct = 0, 0
    missed = 0
    
    for e in [0.01, 0.03, 0.08, 0.1, .3]:
        total_adv = 0
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            for i in range(32):
                input = unnormalize(data[i].cpu().numpy())
                x_adv = fgs(input, target[i].item(), binary_search=False,
                        stepsize=1, epsilon=e) #normalized
                px = np.argmax(fmodel.predictions(normalize(input))) #renormalized input
                # Failure conditions
                if (x_adv is None) or (px != target[i].item()):
                    missed += 1
                    continue
                inter.append(np.argmax(fmodel.predictions(x_adv)))
                assert (inter[-1] != px and inter[-1] != target[i].item())
                adv.append(torch.from_numpy(x_adv))
                y.append(target[i])
            missed = 0
            if adv == []:
                continue
            adv_batch, target_batch = torch.stack(adv).cuda(), torch.stack(y).cuda()
            
            output = model(adv_batch)
            pred = output.data.max(1, keepdim=True)[1]
            correct = pred.eq(target_batch.data.view_as(pred)).long().cpu().sum()
            n_adv = len(target_batch)-correct.item()
            total_adv += n_adv
            adv, y, inter = [], [], []
        #    print ('generated {}/{} adversarials'.format(n_adv, 32))
        print ('{}, total adv: {}/{}'.format(e, total_adv, len(test_loader.dataset)))



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
    else:
        raise NotImplementedError
    return model


def adv_attack(args, path):
    if args.hyper:
        paths = glob(path + '/*.pt')
        path = [x for x in paths if 'hypermnist_0_0.984390625.pt' in x][0]
        hypernet = utils.load_hypernet(path)
        run_adv_hyper(args, hypernet)
    else:
        model = get_network(args)
        path = 'mnist_clf.pt'
        model.load_state_dict(torch.load(path))
        run_adv_model(args, model)
    

if __name__ == '__main__':
    args = load_args()
    args.stat = netdef.nets()[args.net]
    args.shapes = netdef.nets()[args.net]['shapes']
    if args.scratch:
        path = '/scratch/eecs-share/ratzlafn/HyperGAN/'
        if args.hyper:
            path = path +'exp_models'
    else:
        path = mdir+'mnist/{}/'.format(args.net)

    if args.task == 'adv':
        adv_attack(args, path)
    else:
        raise NotImplementedError
