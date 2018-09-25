import torch
import torch.nn.functional as F


def batch_zero_grad(modules):
    for module in modules:
        module.zero_grad()


def batch_update_optim(optimizers):
    for optimizer in optimizers:
        optimizer.step()


def free_params(modules):
    for module in modules:
        for p in module.parameters():
            p.requires_grad = False


def frozen_params(modules):
    for module in modules:
        for p in module.parameters():
            p.requires_grad = False


def pretrain_loss(encoded, noise):
    mean_z = torch.mean(noise, dim=0, keepdim=True)
    mean_e = torch.mean(encoded, dim=0, keepdim=True)
    mean_loss = F.mse_loss(mean_z, mean_e)
    cov_z = torch.matmul((noise-mean_z).transpose(0, 1), noise-mean_z)
    cov_z /= 999
    cov_e = torch.matmul((encoded-mean_e).transpose(0, 1), encoded-mean_e)
    cov_e /= 999
    cov_loss = F.mse_loss(cov_z, cov_e)
    return mean_loss, cov_loss
