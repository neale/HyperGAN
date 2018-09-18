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
import adversary as adv
import foolbox
import attacks
import logging

from torchvision.utils import save_image
from foolbox.criteria import Misclassification
from foolbox.adversarial import Adversarial

foolbox.utils._print_editable()
foolbox_logger = logging.getLogger('foolbox')
logging.disable(logging.CRITICAL);

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
    return acc, loss, model


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


def unnormalize(x):
    x *= .3081
    x += .1307
    return x


def normalize(x):
    x -= .1307
    x /= .3081
    return x


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

# One step attacks, do they transfer, no need to modify attacks here. 

def model_func(model):
    mean = np.array([.1307]).reshape((1, 1, 1))
    std = np.array([.3081]).reshape((1, 1, 1))
    fmodel = foolbox.models.PyTorchModel(model, 
            bounds=(0, 1), num_classes=10, preprocessing=(mean, std))
    return fmodel


def sample_func(hypernet, arch):
    import torch
    import numpy as np
    import torch.distributions.multivariate_normal as N
    def sample_hypernet(hypernet):
        netE, W1, W2, W3 = hypernet
        mean, cov = torch.zeros(300), torch.eye(300)
        D = N.MultivariateNormal(mean, cov)
        z = D.sample((32,)).cuda()
        z.requires_grad = True
        codes = netE(z)
        l1 = W1(codes[0])
        l2 = W2(codes[1])
        l3 = W3(codes[2])
        return l1, l2, l3

    def weights_to_clf(weights, model):
        state = model.state_dict()
        layers = zip(model.lnames, weights)
        for i, (name, params) in enumerate(layers):
            name = name + '.weight'
            loader = state[name]
            state[name] = params.detach()
            model.load_state_dict(state)
        return model
    
    w_batch = sample_hypernet(hypernet)
    rand = np.random.randint(32)
    sample_w = (w_batch[0][rand], w_batch[1][rand], w_batch[2][rand])
    model = weights_to_clf(sample_w, arch)
    model.eval()
    return model


def attack_batch(data, target, fmodel, eps, attack, hypernet, arch):
    missed = 0
    adv_preds, adv, y = [], [], []
    criterion = Misclassification()
    for i in range(len(target)):
        input = unnormalize(data[i].cpu().numpy())
        label = target[i]
        px = np.argmax(fmodel.predictions(input))
        if px != label.item(): #already misclassified
            continue
        adversarial = Adversarial(fmodel, criterion, input, label.item())
        adversarial._set_hypermodel(hypernet, arch, sample_func, model_func)
        x_adv = attack(adversarial, label=None,
                binary_search=False,
                iterations=100,
                stepsize=1,
                epsilon=eps) #normalized
                #epsilons=[eps]) #normalized
        if x_adv is None:  # Failure
            continue
        adv_preds.append(np.argmax(fmodel.predictions(x_adv)))
        px = np.argmax(fmodel.predictions(normalize(input)))
        if (adv_preds[-1] == px) or (adv_preds[-1] == label.item()):
            continue
        adv.append(torch.from_numpy(x_adv))
        y.append(target[i])
    
    if adv == []:
        adv_batch, target_batch = None, None
    else:
        adv_batch = torch.stack(adv).cuda()
        target_batch = torch.stack(y).cuda()
    return adv_batch, target_batch, adv_preds


# we want to estimate performance of a sampled model on adversarials
def run_adv_hyper(args, hypernet):
    arch = get_network(args)
    arch.lnames = args.stat['layer_names']
    model_base, fmodel_base = sample_fmodel(hypernet, arch)
    fgs = foolbox.attacks.HyperBIM(fmodel_base)
    _, test_loader = datagen.load_mnist(args)
    adv, y = [],  []
    for eps in [0.3]:
        total_adv = 0
        acc, _accs, _vars, _stds = [], [], [], []
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            adv_batch, target_batch, _ = attack_batch(
                    data, target, fmodel_base, eps, fgs, hypernet, arch)
            if adv_batch is None:
                continue
            output = model_base(adv_batch)
            pred = output.data.max(1, keepdim=True)[1]
            correct = pred.eq(target_batch.data.view_as(pred)).long().cpu().sum()
            n_adv = len(target_batch) - correct.item()
            total_adv += n_adv
            padv = np.argmax(fmodel_base.predictions(
                adv_batch[0].cpu().numpy()))
            print (adv_batch.shape)
            save_image(adv_batch, './image_{}.png'.format(eps))
            sys.exit(0)
            sample_adv, pred_labels = [], []
            for _ in range(10):
                model, fmodel = sample_fmodel(hypernet, arch) 
                output = model(adv_batch)
                pred = output.data.max(1, keepdim=True)[1]
                correct = pred.eq(target_batch.data.view_as(pred)).long().cpu().sum()
                acc.append(correct.item())
                n_adv_sample = len(target_batch)-correct.item()
                sample_adv.append(n_adv_sample)
                pred_labels.append(pred.view(pred.numel()))

            p_labels = torch.stack(pred_labels).float().transpose(0, 1)
            acc = torch.tensor(acc, dtype=torch.float)
            _accs.append(torch.mean(acc))
            _vars.append(p_labels.var(1).mean())
            _stds.append(p_labels.std(1).mean())
            acc, adv, y = [], [], []

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


def measure_models(m1, m2, i):
    m1_conv = m1['conv1.0.weight'].view(-1)
    m2_conv = m2['conv1.0.weight'].view(-1)
    m1_linear = m1['linear.weight']
    m2_linear = m2['linear.weight']
    
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
    

""" train and save models and their weights """
def run_model_search(args, path):

    for i in range(0, 500):
        print ("\nRunning MNIST Model {}...".format(i))
        model = get_network(args)
        print (model)
        model = w_init(model, 'normal')
        acc, loss, model = train(args, model)
        #extract_weights_all(args, model, i)
        torch.save(model.state_dict(),
                mdir+'mnist/{}/mnist_model_{}_{}.pt'.format(args.net, i, acc))


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
        path = mdir+'mnist/{}/'.format(args.net)

    if args.task == 'test':
        load_models(args, path)
    elif args.task =='train':
        run_model_search(args, path)
    elif args.task == 'adv':
        adv_attack(args, path)
    else:
        raise NotImplementedError
