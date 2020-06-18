import torch
import numpy as np
import torch.nn.functional as F

import utils
import datagen

import os
from scipy.stats import entropy as entropy_fn


def eval_mnist_hypergan(hypergan, ens_size, s_dim, outlier=False):
    hypergan.eval_()
    if outlier is True:
        trainloader, testloader = datagen.load_notmnist()
    else:
        trainloader, testloader = datagen.load_mnist()

    model_outputs = torch.zeros(ens_size, len(testloader.dataset), 10)
    for i, (data, target) in enumerate(testloader):
        data = data.cuda()
        target = target.cuda()
        z = torch.randn(ens_size, s_dim).to(hypergan.device)
        codes = hypergan.mixer(z)
        params = hypergan.generator(codes)
        outputs = []
        for (layers) in zip(*params):
            output = hypergan.eval_f(layers, data)
            outputs.append(output)
        outputs = torch.stack(outputs)
        model_outputs[:, i*len(data):(i+1)*len(data), :] = outputs
    
    # Soft Voting (entropy in confidence)
    probs_soft = F.softmax(model_outputs, dim=-1)  # [ens, data, 10]
    preds_soft = probs_soft.mean(0)  # [data, 10]
    entropy = entropy_fn(preds_soft.T.cpu().numpy()) # [data]

    # Hard Voting (variance in predicted classed)
    probs_hard = F.softmax(model_outputs, dim=-1) #[ens, data, 10]
    preds_hard = probs_hard.var(0).cpu()  # [data, 10]
    variance = preds_hard.sum(1).numpy()  # [data]
    hypergan.train_()

    return entropy, variance


def eval_mnist_ensemble(ensemble, outlier=False):
    for model in ensemble:
        model.eval()

    if outlier is True:
        trainloader, testloader = datagen.load_notmnist()
    else:
        trainloader, testloader = datagen.load_mnist()

    model_outputs = torch.zeros(len(ensemble), len(testloader.dataset), 10)
    for i, (data, target) in enumerate(testloader):
        data = data.cuda()
        target = target.cuda()
        outputs = []
        for model in ensemble:
            outputs.append(model(data))
        outputs = torch.stack(outputs)
        model_outputs[:, i*len(data):(i+1)*len(data), :] = outputs

    # Soft Voting (entropy in confidence)
    probs_soft = F.softmax(model_outputs, dim=-1)  # [ens, data, 10]
    preds_soft = probs_soft.mean(0)  # [data, 10]
    entropy = entropy_fn(preds_soft.T.cpu().numpy()) # [data]

    # Hard Voting (variance in predicted classed)
    probs_hard = F.softmax(model_outputs, dim=-1) #[ens, data, 10]
    preds_hard = probs_hard.var(0).cpu()  # [data, 10]
    variance = preds_hard.sum(1).numpy()  # [data]
    for model in ensemble:
        model.train()

    return entropy, variance


def eval_cifar5_hypergan(hypergan, ens_size, s_dim, outlier=False):
    hypergan.eval_()
    if outlier is True:
        trainloader, testloader = datagen.load_cifar10()
    else:
        trainloader, testloader = datagen.load_cifar10()

    model_outputs = torch.zeros(ens_size, len(testloader.dataset), 10)
    model_outputs = torch.zeros(ens_size, len(testloader.dataset), 5)
    for i, (data, target) in enumerate(testloader):
        data = data.cuda()
        target = target.cuda()
        z = torch.randn(ens_size, s_dim).to(hypergan.device)
        codes = hypergan.mixer(z)
        params = hypergan.generator(codes)
        outputs = []
        for (layers) in zip(*params):
            output = hypergan.eval_f(layers, data)
            outputs.append(output)
        outputs = torch.stack(outputs)
        model_outputs[:, i*len(data):(i+1)*len(data), :] = outputs

    # Soft Voting (entropy in confidence)
    probs_soft = F.softmax(outputs, dim=-1)  # [ens, data, 10]
    preds_soft = probs_soft.mean(0)  # [data, 10]
    entropy = entropy_fn(preds_soft.T.cpu().numpy())

    # Hard Voting (variance in predicted classed)
    probs_hard = F.softmax(outputs, dim=-1) #[ens, data, 10]
    preds_hard = probs_hard.argmax(-1).cpu()  # [ens, data, 1]
    variance = preds_hard.var(0)  # [data, 1]
    hypergan.train_()

    return entropy, variance


def eval_cifar5_ensemble(ensemble, outlier=False):
    for model in ensemble:
        model.eval()

    if outlier is True:
        trainloader, testloader = datagen.load_cifar5()
    else:
        trainloader, testloader = datagen.load_cifar5()

    model_outputs = torch.zeros(len(ensemble), len(testloader.dataset), 10)
    for i, (data, target) in enumerate(testloader):
        data = data.cuda()
        target = target.cuda()
        outputs = []
        for model in ensemble:
            outputs.append(model(data))
        outputs = torch.stack(outputs)
        model_outputs[:, i*len(data):(i+1)*len(data), :] = outputs

    # Soft Voting (entropy in confidence)
    probs_soft = F.softmax(model_outputs, dim=-1)  # [ens, data, 10]
    preds_soft = probs_soft.mean(0)  # [data, 10]
    entropy = entropy_fn(preds_soft.T.cpu().numpy()) # [data]

    # Hard Voting (variance in predicted classed)
    probs_hard = F.softmax(model_outputs, dim=-1) #[ens, data, 10]
    preds_hard = probs_hard.var(0).cpu()  # [data, 10]
    variance = preds_hard.sum(1).numpy()  # [data]
    for model in ensemble:
        model.train()

    return entropy, variance


