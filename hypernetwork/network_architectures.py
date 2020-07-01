# Neale Ratzlaff
#
""" Target network architecture definitions
"""
from collections import OrderedDict, namedtuple
import torch.nn.functional as F

architectures = {
        'lenet': OrderedDict([
            ('conv_1', (1, 6, 5, 1, 2)),
            ('conv_2', (6, 16, 5, 1, 0)),
            ('linear_3', (400, 120)),
            ('linear_4', (120, 84)),
            ('linear_5', (84, 10)),
            ('activation', F.relu),
            ('pooling', (F.max_pool2d, 2, 2, (1, 2))),
            ('flatten', (2,)),
            ('n_layers', 5),
            ('name', 'lenet')]
        ),

        'cifar_small': OrderedDict([
            ('conv_1', (3, 32, 3, 1, 0)), 
            ('conv_2', (32, 64, 3, 1, 0)),
            ('conv_3', (64, 64, 3, 1, 0)),
            ('linear_4', (256, 128)),
            ('linear_5', (128, 10)),
            ('activation', F.relu),
            ('pooling', (F.max_pool2d, 2, 2, (1, 2, 3))),
            ('flatten', (3,)),
            ('n_layers', 5),
            ('name', 'cifar-small')]
        )
}

def fetch_architecture(name='lenet', format=True):
    assert (name in architectures), ("Invalid network definition key " \
            "{}".format(name))
    
    arch = architectures[name]
    arch_container = namedtuple('target_arch', arch.keys())
    return arch_container(**arch)


def fetch_network_keys():
    return ['lenet', 'cifar_small']

