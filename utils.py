import torch
import torch.nn as nn
import torch.nn.init as init
import torch.distributions.multivariate_normal as N
import torch.distributions.uniform as U
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

import math
import numpy as np
from bisect import bisect_right,bisect_left


def sample_hypernet_mnist(args ,hypernet, num):
    netE, W1, W2, W3 = hypernet
    x_dist = create_d(args.ze)
    z = sample_d(x_dist, num)
    codes = netE(z)
    l1 = W1(codes[0])
    l2 = W2(codes[1])
    l3 = W3(codes[2])
    return l1, l2, l3, codes


def sample_hypernet_cifar(args, hypernet, num):
    netE, W1, W2, W3, W4, W5 = hypernet
    x_dist = create_d(args.ze)
    z = sample_d(x_dist, num)
    codes = netE(z)
    l1 = W1(codes[0])
    l2 = W2(codes[1])
    l3 = W3(codes[2])
    l4 = W4(codes[3])
    l5 = W5(codes[4])
    return l1, l2, l3, l4, l5, codes


def weights_to_clf(weights, model, names):
    state = model.state_dict()
    layers = zip(names, weights)
    for i, (name, params) in enumerate(layers):
        name = name + '.weight'
        loader = state[name]
        state[name] = params.detach()
        model.load_state_dict(state)
    return model


class CyclicCosAnnealingLR(_LRScheduler):
    def __init__(self, optimizer,milestones, eta_min=0, last_epoch=-1):
        self.eta_min = eta_min
        self.milestones=milestones
        super(CyclicCosAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.milestones[-1]:
            return [self.eta_min for base_lr in self.base_lrs]
        idx = bisect_right(self.milestones,self.last_epoch)
        left_barrier = 0 if idx==0 else self.milestones[idx-1]
        right_barrier = self.milestones[idx]
        width = right_barrier - left_barrier
        curr_pos = self.last_epoch- left_barrier 
        return [self.eta_min + (base_lr - self.eta_min) *
               (1 + math.cos(math.pi * curr_pos/ width)) / 2
                for base_lr in self.base_lrs]


def batch_rbf_xy(x, y, h_min=1e-3):
    """
	xs(`tf.Tensor`): A tensor of shape (N x Kx x D) containing N sets of Kx
	    particles of dimension D. This is the first kernel argument.
	ys(`tf.Tensor`): A tensor of shape (N x Ky x D) containing N sets of Kx
	    particles of dimension D. This is the second kernel argument.
	h_min(`float`): Minimum bandwidth.
    """
    Kx, D = x.shape[-2:]
    Ky, D2 = y.shape[-2:]
    assert D == D2
    leading_shape = x.shape[:-2]
    diff = x.unsqueeze(-2) - y.unsqueeze(-3)
    # ... x Kx x Ky x D
    dist_sq = torch.sum(diff**2, -1)
    input_shape = (*leading_shape, *[Kx * Ky])
    values, _ = torch.topk(dist_sq.view(*input_shape), k=(Kx * Ky // 2 + 1))  # ... x floor(Ks*Kd/2)

    medians_sq = values[..., -1]  # ... (shape) (last element is the median)
    h = medians_sq / np.log(Kx)  # ... (shape)
    h = torch.max(h, torch.tensor([h_min]).cuda())
    h = h.detach()  # Just in case.
    h_expanded_twice = h.unsqueeze(-1).unsqueeze(-1)
    # ... x 1 x 1
    kappa = torch.exp(-dist_sq / h_expanded_twice)  # ... x Kx x Ky
    # Construct the gradient
    h_expanded_thrice = h_expanded_twice.unsqueeze(-1)
    # ... x 1 x 1 x 1
    kappa_expanded = kappa.unsqueeze(-1)  # ... x Kx x Ky x 1

    kappa_grad = -2 * diff / h_expanded_thrice * kappa_expanded
    # ... x Kx x Ky x D
    return kappa, kappa_grad

def batch_rbf(x, y, h_min=1e-3):
    """
        x (tensor): A tensor of shape (Nx, B, D) containing Nx particles
        y (tensor): A tensor of shape (Ny, B, D) containing Ny particles
        h_min(`float`): Minimum bandwidth.
    """
    Nx, Bx, Dx = x.shape 
    Ny, By, Dy = y.shape
    assert Bx == By
    assert Dx == Dy
    diff = x.unsqueeze(1) - y.unsqueeze(0) # Nx x Ny x B x D
    dist_sq = torch.sum(diff**2, -1).mean(dim=-1) # Nx x Ny
    values, _ = torch.topk(dist_sq.view(-1), k=dist_sq.nelement()//2+1)
    median_sq = values[-1]
    h = median_sq / np.log(Nx)
    h = torch.max(h, torch.tensor([h_min]).cuda())
    # Nx x Ny
    kappa = torch.exp(-dist_sq / h)
    # Nx x Ny x B x D
    kappa_grad = torch.einsum('ij,ijkl->ijkl', kappa, -2 * diff / h)

    return kappa, kappa_grad


