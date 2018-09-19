import torch
import torch.nn as nn
import torch.nn.init as init
import torch.distributions.multivariate_normal as N


def create_d(shape):
    mean = torch.zeros(shape)
    cov = torch.eye(shape)
    D = N.MultivariateNormal(mean, cov)
    return D

def sample_d(D, shape, scale=1., grad=True):
    z = scale * D.sample((shape,)).cuda()
    z.requires_grad = grad
    return z

def sample_hypernet(hypernet, nz):
    netE, W1, W2, W3 = hypernet
    x_dist = create_d(300)
    z = sample_d(x_dist, 32)
    codes = netE(z)
    l1 = W1(codes[0])
    l2 = W2(codes[1])
    l3 = W3(codes[2])
    return l1, l2, l3


def weights_to_clf(weights, model, names):
    state = model.state_dict()
    layers = zip(names, weights)
    for i, (name, params) in enumerate(layers):
        name = name + '.weight'
        loader = state[name]
        state[name] = params.detach()
        model.load_state_dict(state)
    return model
