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
from scipy.stats import mode
from scipy.stats import entropy

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

import statsmodels.api as sm

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
    parser.add_argument('--exp', type=str, default='0', metavar='N', help='')

    args = parser.parse_args()
    return args


def plot_empirical_cdf(args, a, n):
    a = np.asarray(a).reshape(-1)
    np.save('/scratch/eecs-share/ratzlafn/{}_{}_ent_data_{}.npy'.format(
        args.dataset, n, args.exp), a)
    ecdf = sm.distributions.ECDF(a)
    x = np.linspace(min(a), max(a))
    y = ecdf(x)
    np.save('notmnist_cdfx.npy', x)
    np.save('notmnist_cdfy.npy', y)
    plt.plot(x, y)
    print ('ecdf with {} samples'.format(len(a)))
    plt.savefig('/scratch/eecs-share/ratzlafn/{}_{}_ent_{}.png'.format(
        args.dataset, n, args.exp))


# Basically I need to modify an attack so that it takes the
# gradient under a new network for each iteration: IFGSM
# One step attacks, do they transfer, no need to modify attacks here. 
def sample_fmodel(args, hypernet, arch):
    w_batch = utils.sample_hypernet(hypernet)
    rand = np.random.randint(32)
    sample_w = (w_batch[0][rand], w_batch[1][rand], w_batch[2][rand])
    model = utils.weights_to_clf(sample_w, arch, args.stat['layer_names'])
    return model


def unnormalize(x):
    x *= .3081
    x += .1307
    return x


def normalize(x):
    x -= .1307
    x /= .3081
    return x


class FusedNet(nn.Module):
    def __init__(self, networks):
        super(FusedNet, self).__init__()
        self.networks = networks

    def forward(self, x):
        logits = []
        for network in self.networks:
            logits.append(network(x))
        logits = torch.stack(logits).mean(0)
        return logits


def run_anomaly_omni(args, hypernet):
    arch = get_network(args)
    omni_loader = datagen.load_omniglot(args)
    _vars, _stds, _ents = [], [], []
    model = sample_fmodel(args, hypernet, arch) 
    for n_models in [100, 1000]:
        print (n_models)
        _ents = []
        for idx, (data, target) in enumerate(omni_loader):
            data, target = data.cuda(), target.cuda()
            logits, softs = [], []
            with torch.no_grad():
                for _ in range(n_models):
                    model = sample_fmodel(args, hypernet, arch) 
                    output = model(data)
                    softs.append(F.softmax(output, dim=1))
            _softs = torch.stack(softs).float()
            ent = float(entropy(_softs.mean(0).transpose(0, 1).detach()).mean())
            _ents.append(ent)

        plot_empirical_cdf(args, _ents, n_models)
        print ('mean ent: {}'.format(torch.tensor(_ents).mean()))


def run_anomaly_mnist(args, hypernet):
    arch = get_network(args)
    train, test  = datagen.load_mnist(args)
    _vars, _stds, _ents = [], [], []
    model = sample_fmodel(args, hypernet, arch) 
    for n_models in [10, 20, 30]:
        _ents = []
        for idx, (data, target) in enumerate(train):
            data, target = data.cuda(), target.cuda()
            pred_labels = []
            logits = []
            softs = []
            with torch.no_grad():
                for _ in range(n_models):
                    model = sample_fmodel(args, hypernet, arch) 
                    output = model(data)
                    softs.append(F.softmax(output, 1))
            _softs = torch.stack(softs).float()
            std1 = _softs.std(0)
            std2, std3, std4 = std1*2, std1*3, std1*4
            mean_softmax = _softs.mean(0)
            ent0 = float(entropy((mean_softmax).transpose(0, 1).detach()).mean())
            ent4 = float(entropy((mean_softmax + std4).transpose(0, 1).detach()).mean())
            _ents.append(ent0)
            _ents.append(ent4)
            
        plot_empirical_cdf(args, _ents, n_models)
        print ('mean ent: {}'.format(torch.tensor(_ents).mean()))


def run_anomaly_notmnist(args, hypernet):
    arch = get_network(args)
    train, test  = datagen.load_notmnist(args)
    _vars, _stds, _ents = [], [], []
    model = sample_fmodel(args, hypernet, arch) 
    for n_models in [10, 20, 30]:
        _ents = []
        for idx, (data, target) in enumerate(train):
            data, target = data.cuda(), target.cuda()
            pred_labels = []
            logits = []
            softs = []
            with torch.no_grad():
                for _ in range(n_models):
                    model = sample_fmodel(args, hypernet, arch) 
                    output = model(data)
                    softs.append(F.softmax(output, 1))
            _softs = torch.stack(softs).float()
            std1 = _softs.std(0)
            std2, std3, std4 = std1*2, std1*3, std1*4
            mean_softmax = _softs.mean(0)
            ent0 = float(entropy((mean_softmax).transpose(0, 1).detach()).mean())
            #ent1 = float(entropy((mean_softmax + std1).transpose(0, 1).detach()).mean())
            #ent2 = float(entropy((mean_softmax + std2).transpose(0, 1).detach()).mean())
            ent4 = float(entropy((mean_softmax + std4).transpose(0, 1).detach()).mean())
            #ent = float(entropy(_softs.mean(0).transpose(0, 1).detach()).mean())
            _ents.append(ent0)
            _ents.append(ent4)
            
        plot_empirical_cdf(args, _ents, n_models)
        print ('mean ent: {}'.format(torch.tensor(_ents).mean()))

 
def run_anomaly_notmnist_model(args, models):
    arch = get_network(args)
    train, test  = datagen.load_notmnist(args)
    _vars, _stds, _ents = [], [], []
    _ents = []
    for idx, (data, target) in enumerate(train):
        data, target = data.cuda(), target.cuda()
        pred_labels = []
        logits = []
        softs = []
        for i in range(len(models)):
            output = models[i](data)
            softs.append(F.softmax(output, 1))
        _softs = torch.stack(softs).float()
        std1 = _softs.std(0)
        std2, std3, std4 = std1*2, std1*3, std1*4
        mean_softmax = _softs.mean(0)
        ent0 = float(entropy((mean_softmax).transpose(0, 1).detach()).mean())
        #ent1 = float(entropy((mean_softmax + std1).transpose(0, 1).detach()).mean())
        #ent2 = float(entropy((mean_softmax + std2).transpose(0, 1).detach()).mean())
        #ent4 = float(entropy((mean_softmax + std4).transpose(0, 1).detach()).mean())
        #ent = float(entropy(_softs.mean(0).transpose(0, 1).detach()).mean())
        _ents.append(ent0)
        #_ents.append(ent4)
        
    plot_empirical_cdf(args, _ents, len(models))
    print ('mean ent: {}'.format(torch.tensor(_ents).mean()))

       
def run_adv_model(args, models):
    for model in models:
        model.eval()
    print ('models loaded')
    #models = models[:5]
    model = FusedNet(models)
    print ('made fusednet')
    fmodel = attacks.load_model(model)
    criterion = Misclassification()
    fgs = foolbox.attacks.FGSM(fmodel)
    print ('created attack')
    _, test_loader = datagen.load_mnist(args)
    print ('loaded dataset')
    for eps in [0.01, 0.03, 0.08, .1, .3, .5, 1.0]:
        total_adv = 0
        _soft, _logs, _vars, _ents, _lsoft = [], [], [], [], []
        _soft_adv, _logs_adv, _vars_adv, _ents_adv, _lsoft_adv = [], [], [], [], []
        _kl_real, _kl_adv = [], []
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            adv_batch, target_batch, _ = sample_adv_batch(data, target, fmodel, eps, fgs)
            
            if adv_batch is None:
                continue
            # get intial prediction of ensemble, sure
            output = model(adv_batch)
            pred = output.data.max(1, keepdim=True)[1]
            correct = pred.eq(target_batch.data.view_as(pred)).long().cpu().sum()
            n_adv = len(target_batch)-correct.item()
            
            # set up to sample from individual models
            soft_out, pred_out, logits, lsoft_out = [], [], [], []
            soft_out_adv, pred_out_adv, logits_adv, lsoft_out_adv = [], [], [], []

            for i in range(len(models)):
                output = models[i](data)
                soft_out.append(F.softmax(output, dim=1))
                pred_out.append(output.data.max(1, keepdim=True)[1])
                lsoft_out.append(F.log_softmax(output, dim=1))
                logits.append(output)
                
                output = models[i](adv_batch)
                soft_out_adv.append(F.softmax(output, dim=1))
                lsoft_out_adv.append(F.log_softmax(output, dim=1))

            softs = torch.stack(soft_out).float()
            preds = torch.stack(pred_out).float()
            lsoft = torch.stack(lsoft_out).float()
            logs = torch.stack(logits).float()
            softs_adv = torch.stack(soft_out_adv).float()
            lsoft_adv = torch.stack(lsoft_out_adv).float()
            
            # Measure variance of individual logits across models. 
            # HyperGAN ensemble has lower variance across 10 class predictions 
            # But a single logit has high variance acorss models
            units_softmax = softs.var(0).mean().item() # var across models across images
            units_logprob = logs.var(0).mean().item()
            ensemble_var = softs.mean(0).var(1).mean().item()  
            ent = float(entropy(softs.mean(0).transpose(0, 1).detach()).mean())
            #units_logprob = logs.var(0).mean().item()
            ent_adv = float(entropy(softs_adv.mean(0).transpose(0, 1).detach()).mean())
            units_softmax_adv = softs_adv.var(0).mean().item() # var across models - images
            ensemble_var_adv = softs_adv.mean(0).var(1).mean().item()
            
            """ Core Debug """
            # print ('softmax var: ', units_softmax)
            # print ('logprob var: ', units_logprob)
            # print ('ensemble var: ', ensemble_var)
            
            # build lists
            _soft.append(units_softmax)
            _soft_adv.append(units_softmax_adv)
            
            _logs.append(units_logprob)
            # log variance
            _ents.append(ent)
            _ents_adv.append(ent_adv)

            total_adv += n_adv
            if idx > 10:
                print ('REAL: ent: {}'.format(torch.tensor(_ents).mean()))
                print ('ADV Eps: {}, ent: {}'.format(
                    eps, torch.tensor(_ents_adv).mean()))
                break;




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



def run_anomaly(args, path):
    if args.hyper:
        paths = glob(path + '/*.pt')
        path = [x for x in paths if 'hypermnist_0_0.984390625.pt' in x][0]
        path = './hypermnist_disc_0.8857125.pt'
        path = './hypermnist_disc_0.900834375.pt'
        hypernet = utils.load_hypernet(path)
        print ('running on {}'.format(args.dataset))
        if args.dataset == 'omniglot':
            run_anomaly_omni(args, hypernet)
        elif args.dataset == 'notmnist':
            run_anomaly_notmnist(args, hypernet)
        elif args.dataset == 'mnist':
            run_anomaly_mnist(args, hypernet)
        else:
            raise NotImplementedError
    else:
        paths = ['mnist_model_small2_0.pt',
                 'mnist_model_small2_1.pt',
                 #'mnist_model_small2_2.pt',
                 #'mnist_model_small2_3.pt',
                 #'mnist_model_small2_4.pt',
                 #'mnist_model_small2_5.pt',
                 #'mnist_model_small2_6.pt',
                 #'mnist_model_small2_7.pt',
                 #'mnist_model_small2_8.pt',
                 #'mnist_model_small2_9.pt'
                 ]
        models = []
        for path in paths:
            model = get_network(args)
            model.load_state_dict(torch.load(path))
            models.append(model.eval())
        run_anomaly_notmnist_model(args, models)

def adv_attack(args, path):
    if args.hyper:
        paths = glob(path + '/*.pt')
        path = [x for x in paths if 'hypermnist_mean50_0.9866.pt' in x][0]
        path = './hypermnist_0_0.984871875.pt'
        #path = './hypermnist_disc_0.97311875.pt'
        #path = './hypermnist_disc_0.97581875.pt'
        path = './hypermnist_disc_0.8857125.pt'
        hypernet = utils.load_hypernet(path)
        run_adv_hyper(args, hypernet)

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

    if args.task == 'anomaly':
        run_anomaly(args, path)
    else:
        raise NotImplementedError
