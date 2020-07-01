# Neale Ratzlaff
# 
""" evaluate_uncertinaty.py
"""

import os
import torch
import numpy as np
import torch.nn.functional as F
from scipy.stats import entropy as entropy_fn

import utils
import datagen


def eval_mnist_hypernetwork(hypernetwork, ens_size, s_dim, device, outlier=False):
    hypernetwork.eval()
    if outlier is True:
        trainloader, testloader = datagen.load_notmnist()
    else:
        trainloader, testloader = datagen.load_mnist()

    model_outputs = torch.zeros(ens_size, len(testloader.dataset), 10)
    for i, (data, target) in enumerate(testloader):
        data = data.to(device)
        target = target.to(device)
        with torch.no_grad():
            z = hypernetwork.sample_generator_input()
            theta = hypernetwork.sample_parameters(z)
            hypernetwork.set_parameters_to_model(theta)
            outputs = hypernetwork.forward_model(data)
        outputs = outputs.transpose(0, 1) # [B, N, D] -> [N, B, D]
        outputs = outputs[:ens_size]
        model_outputs[:, i*len(data):(i+1)*len(data), :] = outputs
    
    # Soft Voting (entropy in confidence)
    probs_soft = F.softmax(model_outputs, dim=-1)  # [ens, data, 10]
    preds_soft = probs_soft.mean(0)  # [data, 10]
    entropy = entropy_fn(preds_soft.T.cpu().numpy()) # [data]

    # Hard Voting (variance in predicted classed)
    probs_hard = F.softmax(model_outputs, dim=-1) #[ens, data, 10]
    preds_hard = probs_hard.var(0).cpu()  # [data, 10]
    variance = preds_hard.sum(1).numpy()  # [data]
    hypernetwork.train()

    return entropy, variance


def eval_mnist_ensemble(ensemble, device, outlier=False):
    for model in ensemble:
        model.eval()

    if outlier is True:
        trainloader, testloader = datagen.load_notmnist()
    else:
        trainloader, testloader = datagen.load_mnist()

    model_outputs = torch.zeros(len(ensemble), len(testloader.dataset), 10)
    for i, (data, target) in enumerate(testloader):
        data = data.to(device)
        target = target.to(device)
        with torch.no_grad():
            outputs = self.predict(data)
        outputs = outputs.transpose(0, 1) # [B, N, D] -> [N, B, D]
        outputs = outputs[:ens_size]
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


def eval_cifar5_hypernetwork(hypernetwork, ens_size, s_dim, device, outlier=False):
    hypernetwork.eval()
    if outlier is True:
        trainloader, testloader = datagen.load_cifar10()
    else:
        trainloader, testloader = datagen.load_cifar10()

    model_outputs = torch.zeros(ens_size, len(testloader.dataset), 10)
    model_outputs = torch.zeros(ens_size, len(testloader.dataset), 5)
    for i, (data, target) in enumerate(testloader):
        data = data.to(device)
        target = target.to(device)
        with torch.no_grad():
            z = hypernetwork.sample_generator_input()
            theta = hypernetwork.sample_parameters(z)
            hypernetwork.set_parameters_to_model(theta)
            outputs = hypernetwork.forward_model(data)
        outputs = outputs.transpose(0, 1) # [B, N, D] -> [N, B, D]
        model_outputs[:, i*len(data):(i+1)*len(data), :] = outputs

    # Soft Voting (entropy in confidence)
    probs_soft = F.softmax(outputs, dim=-1)  # [ens, data, 10]
    preds_soft = probs_soft.mean(0)  # [data, 10]
    entropy = entropy_fn(preds_soft.T.cpu().numpy())

    # Hard Voting (variance in predicted classed)
    probs_hard = F.softmax(outputs, dim=-1) #[ens, data, 10]
    preds_hard = probs_hard.argmax(-1).cpu()  # [ens, data, 1]
    variance = preds_hard.var(0)  # [data, 1]
    hypernetwork.train()

    return entropy, variance


def eval_cifar5_ensemble(ensemble, device, outlier=False):
    for model in ensemble:
        model.eval()

    if outlier is True:
        trainloader, testloader = datagen.load_cifar5()
    else:
        trainloader, testloader = datagen.load_cifar5()

    model_outputs = torch.zeros(len(ensemble), len(testloader.dataset), 10)
    for i, (data, target) in enumerate(testloader):
        data = data.to(device)
        target = target.to(device)
        with torch.no_grad():
            outputs = self.predict(data)
        outputs = outputs.transpose(0, 1) # [B, N, D] -> [N, B, D]
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

