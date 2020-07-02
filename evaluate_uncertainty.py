# Neale Ratzlaff
# 
""" evaluate_uncertinaty.py
"""

import os
import torch
import numpy as np
import torch.nn.functional as F
from scipy.stats import entropy as entropy_fn
from sklearn.metrics import roc_auc_score
import utils
import datagen


def predict(train, test, model, size, device):
    try:
        n_cls = len(test.dataset.classes)
    except:
        n_cls = len(test.dataset.dataset.classes)
    model_outputs = torch.zeros(size, len(test.dataset), 10)
    for i, (data, target) in enumerate(test):
        data = data.to(device)
        target = target.to(device)
        with torch.no_grad():
            z = model.sample_generator_input()
            theta = model.sample_parameters(z)
            model.set_parameters_to_model(theta)
            outputs = model.forward_model(data)
        outputs = outputs.transpose(0, 1) # [B, N, D] -> [N, B, D]
        outputs = outputs[:size]
        model_outputs[:, i*len(data):(i+1)*len(data), :] = outputs
    return model_outputs


def uncertainty(outputs): 
    # Soft Voting (entropy in confidence)
    probs_soft = F.softmax(outputs, dim=-1)  # [ens, data, 10]
    preds_soft = probs_soft.mean(0)  # [data, 10]
    entropy = entropy_fn(preds_soft.T.cpu().numpy()) # [data]
    
    # Hard Voting (variance in predicted classed)
    probs_hard = F.softmax(outputs, dim=-1) #[ens, data, 10]
    preds_hard = probs_hard.var(0).cpu()  # [data, 10]
    variance = preds_hard.sum(1).numpy()  # [data]
    return (entropy, variance)


def eval_mnist_hypernetwork(hypernetwork, ens_size, s_dim, device):
    hypernetwork.eval()
    
    mnist_train, mnist_test = datagen.load_mnist()
    outputs_mnist = predict(mnist_train, mnist_test, hypernetwork, ens_size, device)
    uncertainty_mnist = uncertainty(outputs_mnist)
    
    notmnist_train, notmnist_test = datagen.load_notmnist()
    outputs_notmnist = predict(notmnist_train, notmnist_test, hypernetwork, ens_size, device)
    uncertainty_notmnist = uncertainty(outputs_notmnist)
    
    y_true = np.array([0] * len(mnist_test.dataset) + [1] * len(notmnist_test.dataset))
    y_score = np.concatenate([uncertainty_mnist[0], uncertainty_notmnist[0]])

    auc_score = roc_auc_score(y_true, y_score)
    
    hypernetwork.train()
    return uncertainty_mnist, uncertainty_notmnist, auc_score


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


def eval_cifarx_hypernetwork(hypernetwork, ens_size, s_dim, device, outlier=False):
    hypernetwork.eval()
    cifar5a_train, cifar5a_test = datagen.load_cifarx(c_idx=[0,1,2,3,4])
    cifar5b_train, cifar5b_test = datagen.load_cifarx(c_idx=[5,6,7,8,9])
    
    outputs_cifar5a = predict(cifar5a_train, cifar5a_test, hypernetwork, ens_size, device)
    uncertainty_cifar5a = uncertainty(outputs_cifar5a)

    outputs_cifar5b = predict(cifar5b_train, cifar5b_test, hypernetwork, ens_size, device)
    uncertainty_cifar5b = uncertainty(outputs_cifar5b)

    y_true = np.array([0] * len(cifar5a_test.dataset) + [1] * len(cifar5b_test.dataset))
    y_score = np.concatenate([uncertainty_cifar5a[0], uncertainty_cifar5b[0]])

    auc_score = roc_auc_score(y_true, y_score)

    hypernetwork.train()
    return uncertainty_cifar5a, uncertainty_cifar5b, auc_score


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

