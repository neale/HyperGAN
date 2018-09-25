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


# Basically I need to modify an attack so that it takes the
# gradient under a new network for each iteration: IFGSM
# One step attacks, do they transfer, no need to modify attacks here. 
def sample_fmodel(args, hypernet, arch):
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


def sample_adv_batch(data, target, fmodel, eps, attack):
    missed = 0
    inter, adv, y = [], [], []

    for i in range(32):
        input = unnormalize(data[i].cpu().numpy())
        x_adv = attack(input, target[i].item(),
                binary_search=False,
                stepsize=1,
                epsilon=eps) #normalized
                #epsilons=[eps]) #normalized
        px = np.argmax(fmodel.predictions(normalize(input))) #renormalized input
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
    models, fmodels = [], []
    #for i in range(10):
    #    model_base, fmodel_base = sample_fmodel(args, hypernet, arch)
    #    models.append(model_base)
    #    fmodels.append(fmodel_base)   
    #fmodel_base = attacks.load_model(FusedNet(models))
    model_base, fmodel_base = sample_fmodel(args, hypernet, arch)
    criterion = Misclassification()
    fgs = foolbox.attacks.BIM(fmodel_base, criterion)
    _, test_loader = datagen.load_mnist(args)
    adv, y = [],  []
    for n_models in [5, 10, 100, 500]:
        print ('ensemble of {}'.format(n_models))
        for eps in [0.01, 0.03, 0.08, 0.1, 0.3, 0.5, 1.0]:
            total_adv = 0
            acc, _accs = [], []
            _kl_real, _kl_adv = [], []
            _soft, _logs, _vars, _ents, _lsoft = [], [], [], [], []
            _soft_adv, _logs_adv, _vars_adv, _ents_adv, _lsoft_adv = [], [], [], [], []
            for idx, (data, target) in enumerate(test_loader):
                data, target = data.cuda(), target.cuda()
                adv_batch, target_batch, _ = sample_adv_batch(
                        data, target, fmodel_base, eps, fgs)
                if adv_batch is None:
                    continue
                # get base hypermodel output, I guess
                output = model_base(adv_batch)
                pred = output.data.max(1, keepdim=True)[1]
                correct = pred.eq(target_batch.data.view_as(pred)).long().cpu().sum()
                n_adv = len(target_batch) - correct.item()
                total_adv += n_adv
               
                soft_out, pred_out, logits, lsoft_out = [], [], [], []
                soft_out_adv, pred_out_adv, logits_adv, lsoft_out_adv = [], [], [], []
                with torch.no_grad():
                    for n in range(n_models):
                        model, fmodel = sample_fmodel(args, hypernet, arch) 
                        output = model(data)
                        soft_out.append(F.softmax(output, dim=1))
                        lsoft_out.append(F.log_softmax(output, dim=1))
                        #pred_out.append(output.data.max(1, keepdim=True)[1])
                        logits.append(output)

                        output = model(adv_batch)
                        soft_out_adv.append(F.softmax(output, dim=1))
                        lsoft_out_adv.append(F.log_softmax(output, dim=1))
                        #pred_out_adv.append(output.data.max(1, keepdim=True)[1])
                        logits_adv.append(output)
                        ## correction graph
                        #pred = output.data.max(1, keepdim=True)[1]
                        #correct = pred.eq(target_batch.data.view_as(pred)).long().cpu().sum()
                        #c = len(pred_out) - correct.item()
                        #print ('got {} / {} / {}'.format(correct.item(), len(target_batch), 32))
                        #dis.append(correct.item()/n_adv)
                        ##
                    #np.save('/scratch/eecs-share/ratzlafn/acc.npy', np.array(dis))
                    #sys.exit(0)    
                        
                softs = torch.stack(soft_out).float()
                lsoft = torch.stack(lsoft_out).float()
                #preds = torch.stack(pred_out).float()
                logs = torch.stack(logits).float()
                softs_adv = torch.stack(soft_out_adv).float()
                lsoft_adv = torch.stack(lsoft_out_adv).float()
                #preds_adv = torch.stack(pred_out_adv).float()
                logs_adv = torch.stack(logits_adv).float()
                #sys.exit(0)
                for j in range(len(adv_batch)):
                    kl_real, kl_adv = 0, 0
                    kl_real_, kl_adv_ = 0, 0
                    for i in range(n_models):
                        kl_real += F.kl_div(lsoft[i][j], F.softmax(softs.mean(0)[j]))
                        kl_adv += F.kl_div(lsoft_adv[i][j], F.softmax(softs_adv.mean(0)[j]))
                    _kl_real.append(kl_real/n_models)
                    _kl_adv.append(kl_adv/n_models)
                # Measure variance of individual logits across models. 
                # HyperGAN ensemble has lower variance across 10 class predictions 
                # But a single logit has high variance acorss models
                units_softmax = softs.var(0).mean().item() # var across models across images
                ent = float(entropy(softs.mean(0).detach()).mean())
                #units_logprob = logs.var(0).mean().item()
                ensemble_var = softs.mean(0).var(1).mean().item()  

                units_softmax_adv = softs_adv.var(0).mean().item() # var across models - images
                ent_adv = float(entropy(softs_adv.mean(0).detach()).mean())
        
                #units_logprob_adv = logs_adv.var(0).mean().item()
                ensemble_var_adv = softs_adv.mean(0).var(1).mean().item()
                """ Core Debug """
                # print ('softmax var: ', units_softmax)
                # print ('logprob var: ', units_logprob)
                # print ('ensemble var: ', ensemble_var)

                # build lists
                _soft.append(units_softmax)
                #_logs.append(units_logprob)
                _vars.append(ensemble_var)
                _ents.append(ent)
                _soft_adv.append(units_softmax_adv)
                #_logs_adv.append(units_logprob_adv)
                _vars_adv.append(ensemble_var_adv)
                _ents_adv.append(ent_adv)

                if idx > 5:
                    #print ('NAT: Log var: -, Softmax var: {}, Ent: {}, Ens var: {}'.format(
                    #    #torch.tensor(_logs).mean(),
                    #    torch.tensor(_soft).mean(),
                    #    torch.tensor(_ents).mean(),
                    #    torch.tensor(_vars).mean()))
                    print ('ADV Eps: {}'.format(#, Log var: -, Softmax var: {}, Ent: {}, Ens var: {}'.format(
                        eps))#,
                        #torch.tensor(_logs_adv).mean(),
                        #torch.tensor(_soft_adv).mean(),
                        #torch.tensor(_ents_adv).mean(),
                        #torch.tensor(_vars_adv).mean()))
                    print ("real KL: {}, adv KL: {}".format(
                        torch.stack(_kl_real).mean(), 
                        torch.stack(_kl_adv).mean()))
                    break;
            """
            print ('[Final] - Eps: {}, Adv: {}/{}, Log var: {}, Softmax var: {}, Ens var: {}'.format(
                 eps, total_adv, len(test_loader.dataset), 
                 torch.tensor(_logs).mean(),
                 torch.tensor(_soft).mean(),
                 torch.tensor(_vars).mean()))
            """

             
def run_adv_model(args, models):
    for model in models:
        model.eval()
    print ('models loaded')
    #models = models[:5]
    model = FusedNet(models)
    print ('made fusednet')
    fmodel = attacks.load_model(model)
    criterion = Misclassification()
    fgs = foolbox.attacks.BIM(fmodel)
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
                
                output = model(adv_batch)
                soft_out_adv.append(F.softmax(output, dim=1))
                lsoft_out_adv.append(F.log_softmax(output, dim=1))

            softs = torch.stack(soft_out).float()
            preds = torch.stack(pred_out).float()
            lsoft = torch.stack(lsoft_out).float()
            logs = torch.stack(logits).float()
            softs_adv = torch.stack(soft_out_adv).float()
            lsoft_adv = torch.stack(lsoft_out_adv).float()
            
            for j in range(len(adv_batch)):
                kl_real, kl_adv = 0, 0
                for i in range(len(models)):
                    kl_real += F.kl_div(lsoft[i][j], F.softmax(softs.mean(0)[j]))
                    kl_adv += F.kl_div(lsoft_adv[i][j], F.softmax(softs_adv.mean(0)[j]))
                _kl_real.append(kl_real/len(models))
                _kl_adv.append(kl_adv/len(models))
            
            # Measure variance of individual logits across models. 
            # HyperGAN ensemble has lower variance across 10 class predictions 
            # But a single logit has high variance acorss models
            units_softmax = softs.var(0).mean().item() # var across models across images
            units_logprob = logs.var(0).mean().item()
            ensemble_var = softs.mean(0).var(1).mean().item()  
            ent = float(entropy(softs.mean(0).detach()).mean())

            units_softmax_adv = softs_adv.var(0).mean().item() # var across models - images
            ent_adv = float(entropy(softs_adv.mean(0).detach()).mean())
            ensemble_var_adv = softs_adv.mean(0).var(1).mean().item()
            
            """ Core Debug """
            # print ('softmax var: ', units_softmax)
            # print ('logprob var: ', units_logprob)
            # print ('ensemble var: ', ensemble_var)
            
            # build lists
            _soft.append(units_softmax)
            _logs.append(units_logprob)
            _vars.append(ensemble_var)
            _ents.append(ent)
            
            _soft_adv.append(units_softmax_adv)
            _vars_adv.append(ensemble_var_adv)
            _ents_adv.append(ent_adv)


            total_adv += n_adv
            if idx % 10 == 0 and idx > 1:
                #print ('NAT: Log var: {}, Softmax var: {}, Ent var: {}, Ens var: {}'.format(
                #    eps,
                #    torch.tensor(_logs).mean(),
                #    torch.tensor(_soft).mean(),
                #    torch.tensor(_ents).mean(),
                #    torch.tensor(_vars).mean()))
                print ('ADV: Eps: {}'.format(#, Ent var: {}, Softmax var: {}, Ens var: {}'.format(
                    eps))#,
                    #torch.tensor(_ents_adv).mean(),
                    #torch.tensor(_soft_adv).mean(),
                    #torch.tensor(_vars_adv).mean()))

                print ("real KL: {}, adv KL: {}".format(
                    torch.stack(_kl_real).mean(), 
                    torch.stack(_kl_adv).mean()))
                break
        """
        print ('[Final] - Eps: {}, Adv: {}/{}, Log var: {}, Softmax var: {}, Ens var: {}'.format(
            eps, total_adv, len(test_loader.dataset), 
            torch.tensor(_logs).mean(),
            torch.tensor(_soft).mean(),
            torch.tensor(_vars).mean()))
        """



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
        path = [x for x in paths if 'hypermnist_mean50_0.9866.pt' in x][0]
        hypernet = utils.load_hypernet(path)
        run_adv_hyper(args, hypernet)
    else:
        paths = ['mnist_model_small2_0.pt',
                 'mnist_model_small2_1.pt',
                 'mnist_model_small2_2.pt',
                 'mnist_model_small2_3.pt',
                 'mnist_model_small2_4.pt',
                 'mnist_model_small2_5.pt',
                 'mnist_model_small2_6.pt',
                 'mnist_model_small2_7.pt',
                 'mnist_model_small2_8.pt',
                 'mnist_model_small2_9.pt'
                 ]
        models = []
        for path in paths:
            model = get_network(args)
            model.load_state_dict(torch.load(path))
            models.append(model.eval())
        run_adv_model(args, models)
    

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

    if args.task == 'adv':
        adv_attack(args, path)
    else:
        raise NotImplementedError
