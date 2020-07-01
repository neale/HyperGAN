# Neale Ratzlaff
# Structures.py
# 
"""Generator and Mixer definitions for Hypernetworks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearMixer(nn.Module):
    def __init__(
            self,
            s_dim=512,
            z_dim=256,
            use_bias=True,
            hidden_layers=(100, 100),
            activation=torch.nn.ReLU,
            n_gen=1,
            use_batchnorm=True,
            clear_bn_bias=True,
            last_layer_act=False):
        """
        Generator for Mixer structure
        Args:
            s_dim (Int): the dimensionality of the sample space used
                as input to the mixer (default 512)
            z_dim (Int): the dimensionality of the latent space that the mixer
                projects to (default 256)
            use_bias (Bool): determines if the hidden layers in the mixer
                use a bias term (default True)
            hidden_layers (tuple[int]): a tuple of Ints, representing the width
                of each hidden layer in the mixer (default (100, 100) 
                representing 2 hidden layers of width 100
            activation (torch.nn function): function specifying the activation
                function used in the mixer hidden layers. Must be an instance
                of torch.nn, (default torch.nn.ReLU)
            last_layer_act (Bool): determines if an activation is to be used
                at the last layer
        """

        super(LinearMixer, self).__init__()
        self._s_dim = s_dim
        self._z_dim = z_dim
        self._bias  = use_bias
        self._hidden_layers = hidden_layers
        self._n_gen = n_gen
        self._use_batchnorm = use_batchnorm
        self._clear_bn_bias = clear_bn_bias

        assert (callable(activation)
                or activation is None), ("Activation must be an callable "\
                "function or None, got {}".format(type(activation)))

        self._activation = activation
        self._last_layer_act = last_layer_act
        
        if last_layer_act is False:
            self._last_act = activation
        else:
            self._last_act = nn.Identity
        
        # First layer
        self.generator = nn.ModuleList([
            nn.Linear(self._s_dim, self._hidden_layers[0], bias=self._bias),
            self._activation()
        ])
        # Hidden layers
        for i, size in enumerate(self._hidden_layers[1:]):
            self.generator.extend([
                    nn.Linear(
                        self._hidden_layers[i-1],
                        self._hidden_layers[i],
                        bias=self._bias),
                    self._activation()])

        # Output layer of mixer: (n_gen * z)
        self.generator.extend([
                nn.Linear(
                    self._hidden_layers[-1],
                    self._z_dim * self._n_gen,
                    bias=self._bias),
                self._last_act()])

    def forward(self, x):
        for layer in self.generator:
            x = layer(x)
        return x


class LinearGenerator(nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            z_dim=256,
            use_bias=True,
            hidden_layers=(100, 100),
            activation=torch.nn.ReLU,
            use_batchnorm=True,
            clear_bn_bias=True,
            last_layer_act=False):
        """
        Generator for linear function parameters. Linear function could be
            any set of parameters that we can reshape as inputs for a valid
            torch.nn.functional function. LinearGenerator outputs parameters
            for a weight matrix of size (W_in, W_out). 
        Args:
            input size (Int): the dimensionality of size W_in 
            output_size (Int): the dimensionality of size W_out
            z_dim (Int): the dimensionality of the generator latent space
                [if using the mixer]: dimensionality of mixer output
                [if not using the mixer]: dimensionality of random sample    
                (default 256). 
            use_bias (Bool): determines if the hidden layers in the generator
                use a bias term (default True)
            hidden_layers (tuple[int]): a tuple of Ints, representing the width
                of each hidden layer in the generator (default (100, 100) 
                representing 2 hidden layers of width 100
            activation (torch.nn function): function specifying the activation
                function used in the generator hidden layers. Must be an 
                instance of torch.nn, (default torch.nn.ReLU)
            last_layer_act (Bool): determines if an activation is to be used
                at the last layer
        """       
        super(LinearGenerator, self).__init__()
        self._z_dim = z_dim
        self._bias  = use_bias
        self._hidden_layers = hidden_layers
        self._n_gen = len(hidden_layers)+1
        self._use_batchnorm = use_batchnorm
        self._clear_bn_bias = clear_bn_bias

        assert (callable(activation)
                or activation is None), ("Activation must be an callable "\
                        "function or None, got {}".format(type(activation)))

        self._activation = activation
        self._last_layer_act = last_layer_act

        self._input_dim = input_size
        self._output_dim = output_size

        if last_layer_act is False:
            self._last_act = activation
        else:
            self._last_act = nn.Identity

        # First layer
        if self._use_batchnorm:
            normalize_or_identity = nn.BatchNorm1d(self._hidden_layers[0])
        else:
            normalize_or_identity = nn.Identity()

        self.generator = nn.ModuleList([
            nn.Linear(self._z_dim, self._hidden_layers[0], bias=self._bias),
            normalize_or_identity,
            self._activation()
            ])

        # Hidden layers
        for i, size in enumerate(self._hidden_layers[1:]):
            if self._use_batchnorm:
                normalize_or_identity = nn.BatchNorm1d(self._hidden_layers[i])
            else:
                normalize_or_identity = nn.Identity()
            self.generator.extend([
                nn.Linear(
                    self._hidden_layers[i-1],
                    self._hidden_layers[i],
                    bias=self._bias),
                normalize_or_identity,
                self._activation()])

        # Output layer
        self.generator.extend([
            nn.Linear(
                self._hidden_layers[-1],
                self._output_dim * self._input_dim + self._output_dim,
                bias=self._bias),
            self._last_act()])

    def forward(self, x):
        """ HyperNetwork Generator Core
        inputs:
            x (torch.tensor) [N, z_dim]:
                N : number of particles (number of generated weight matrices)
                if using the mixer, `x` is a `z_dim` dimensional output
                of the mixer, otherwise `x` is a `z_dim` dimensional sample
                from a standard normal distribution
        returns:
            f(x), weight matrix of size [N, input_dim, output_dim]
        """
        for layer in self.generator:
            x = layer(x)
        return x
        
