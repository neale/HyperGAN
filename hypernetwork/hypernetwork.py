# Neale Ratzlaff
# Hypernetwork.py

import torch
import torch.nn.functional as F
from collections import namedtuple
from .hypernetwork_layers import LinearLayer, Conv2DLayer
from .network_architectures import fetch_architecture, fetch_network_keys
from .structures import LinearMixer, LinearGenerator
import warnings


class HyperNetwork(object):
    """HyperNetwork
        
        Design- Force top-level algorithm to be explicit
        Here: implement public functions for sampling noise, sampling params
        Stateless. Parameter/noise storage happens at top level, not here
    """

    def __init__(
            self,
            s_dim=512,
            z_dim=256,
            use_mixer=True,
            noise_type='normal',
            hidden_layers=(100, 100),
            use_batchnorm=True,
            clear_bn_bias=True,
            particles=100,
            use_bias=True,
            activation=torch.nn.ReLU,
            last_layer_act=None,
            target_arch='lenet',
            device=torch.device('cpu'),
            name="HyperNetwork"):
        """
        Args:
            Args for the HyperNetwork
	    Default target network architecture is LeNet-5
            ====================================================================
            use_mixer (Bool): determines if the mixer network is used to
                compress weight embeddings
            hidden_layers (Tuple): size of hidden layers for mixer and
                generator networks
            last_layer_act (nn.functional): activation function of the
                additional layer specified by ``last_layer_size``. Note that if
                ``last_layer_size`` is not None, ``last_activation`` has to be
                specified explicitly.
            target_arch(Str): target network architecture to generate
                parameters for. Choices are [lenet, cifar_small]
            device (torch.device
            Args for the generator
            ====================================================================
            s_dim (Int): dimension of mixer noise
            z_dim (Int): dimension of generator latent space
            noise_type (Str): distribution type of noise input to the generator, 
                types are [``normal``] 
            particles (Int): number of sampling particles
            use_bias (Bool): number of sampling particles
            name (str):
        """

        self._use_mixer = use_mixer
        self._hidden_layers = hidden_layers
        self._last_layer_act = last_layer_act
        self._target_arch = target_arch
        
        self._s_dim = s_dim
        self._z_dim = z_dim
        self._noise_type = noise_type
        self._particles = particles
        self._use_bias = use_bias
        self._activation = activation
        self._use_batchnorm = use_batchnorm
        self._clear_bn_bias = clear_bn_bias
        self._train_loader = None
        self._test_loader = None
        self._regenerate_for_each_batch = True

        self._mixer = None
        self._layer_fns = None
        self._target_net = None
        self._target_dim_map = None
        self._device = device


    def set_data_loader(self, train_loader, test_loader=None):
        self._train_loader = train_loader
        self._test_loader = test_loader

    def set_particles(self, particles):
        self._particles = particles

    @property
    def particles(self):
        return self._particles

    def print_architecture(self):
        if self._mixer is not None:
            print (self._mixer)
        if self._layer_fns is not None:
            print (self._layer_fns)

    def set_target_architecture(self, target=None):
        """
        Setter. pulls the description of the target architecture 
        Sets:
            target_arch_def (namedtuple): describes the target architecture
                completely from use in downstream parameter generation
        """
        if target is None:
            target = self._target_arch
        network_names = fetch_network_keys()
        assert target in network_names, "Target architecture must have a "\
                "full definition, available names: {}, got {}".format(
                        network_names, target)

        self._target_arch_def = fetch_architecture(target)

    def set_hypernetwork_mixer(self):
        """
        Setter. initializes the mixer for the hypernetwork
        Sets: 
            mixer (LinearMixer): mixer structure described in
                (razlafn, fuxin. 2019), https://arxiv.org/pdf/1901.11058.pdf
        """
        self._mixer = LinearMixer(
                s_dim=self._s_dim,
                z_dim=self._z_dim,
                use_bias=self._use_bias,
                hidden_layers=self._hidden_layers,
                activation=self._activation,
                n_gen=self._target_arch_def.n_layers,
                last_layer_act=self._last_layer_act)
        self._mixer.to(self._device)

    def set_hypernetwork_layers(self):
        """ Setter. initializes the generators for each layer of the
                hypernetwork
            Sets:
                n_params (int): total number of parameters generated for target
                    network
                layer_fns (list[LinearGenerator]: initializes and sets the
                    ordering of the hypernetwork layer generators
                network_fns (list[LinearLayer or ConvLayer]: initializes and sets the
                    ordering of the hypernetwork output layers
                target_dim_map (list(tuple)): list of tupes of the form:
                    ((int) # parameters, (str) type: 'conv' or 'linear). 
                    Describes the high level intent of the layer generator 
        """
        assert self._target_arch_def is not None, "Target architecture "\
                "definition must be set before defining generators"
        assert isinstance(self._target_arch_def, tuple), "Target " \
                "architecture definition should be a namedtuple instance, "\
                "got {}".format(type(self._target_arch_def))
        
        pooling = self._target_arch_def.pooling
        activation = self._target_arch_def.activation
        
        total_params = 0
        target_dim_map = []
        layer_fns = []
        network_fns = []
        for name, val in zip(self._target_arch_def._fields, self._target_arch_def):
            if name.startswith('conv'):
                c_in, c_out, kernel, stride, padding = val
                pool_fn, pool_kernel, pool_stride, pool_layers = self._target_arch_def.pooling
                input_size = c_in * kernel * kernel
                output_size = c_out
                target_dim_map.append((input_size*output_size, output_size, 'conv'))
                network_fn = Conv2DLayer(
                        in_channels=c_in,
                        out_channels=c_out,
                        kernel_size=kernel,
                        pooling_fn=pool_fn,
                        pooling_kernel=pool_kernel,
                        pooling_stride=pool_stride,
                        activation=activation,
                        strides=stride,
                        padding=padding,
                        particles=self._particles,
                        use_bias=self._use_bias,
                        device=self._device)

            elif name.startswith('linear'):
                width_in, width_out = val
                input_size = width_in
                output_size = width_out
                target_dim_map.append((input_size*output_size, output_size, 'linear'))
                network_fn = LinearLayer(
                        input_dim=input_size,
                        output_dim=output_size,
                        activation=activation,
                        particles=self._particles,
                        use_bias=self._use_bias,
                        device=self._device)
            else:
                continue

            total_params += input_size * output_size + output_size
            layer = LinearGenerator(
                    input_size=input_size,
                    output_size=output_size,
                    z_dim=self._z_dim,
                    use_bias=self._use_bias,
                    hidden_layers=self._hidden_layers,
                    activation=self._activation,
                    use_batchnorm=self._use_batchnorm,
                    clear_bn_bias=self._clear_bn_bias,
                    last_layer_act=self._last_layer_act)
            layer.to(self._device)
            
            n_params = total_params
            layer_fns.append(layer)
            network_fns.append(network_fn)

        self._n_params = n_params
        self._layer_fns = layer_fns
        self._network_fns = network_fns
        self._target_dim_map = target_dim_map

    def attach_optimizers(self, lr_mixer=None, lr_generator=None, optim=None):
        """ Setter. Initializes optimizers for the generators and 
                Mixer (if in use). Uses AdamW by default but can be
                overridden.
            Args: 
                lr_mixer (float): learning rate for the mixer
                lr_generator (float): learning rate for the generator
                optim_fn (torch.optim): optional override for optimizer 
                    function. For extended optimizer kwargs, pass in a 
                    partial funciton. 
        """
        if optim is None:
            optim_fn = torch.optim.AdamW
        else:
            optim_fn = optim
        if self._mixer is not None and lr_mixer is not None:
            self._optimizer_mixer = optim_fn(
                    self._mixer.parameters(),
                    lr_mixer)
        if self._layer_fns is not None:
            layer_params = []
            for layer in self._layer_fns:
                layer_params += list(layer.parameters())
            self._optimizer_generator = optim_fn(
                    layer_params,
                    lr_generator)
        if self._mixer is None and self._layer_fns is None:
            warnings.warn('Warning. Neither Mixer nor Generators are set, "\
                    "call to attach optimizers returned with no effect')
   

    def zero_grad(self):
        """ zeros the gradients of the mixer and the generator
        """
        if self._mixer is not None:
            self._mixer.zero_grad()
        if self._layer_fns is not None:
            for layer_fn in self._layer_fns:
                layer_fn.zero_grad()
        if self._mixer is None and self._layer_fns is None:
            warnings.warn('Warning. Neither Mixer nor Generators are set, "\
                    "call to zero gradients returned with no effect')
        

    def update_step(self):
        """ Updates the initialized optimizers. 
            This method performs no checks outside of the existence of the 
                optimizers. It is left to the algorithm designers to call 
                the update step correctly. 
        """
        if self._optimizer_mixer is not None:
            self._optimizer_mixer.step()
        
        if self._optimizer_generator is not None:
            self._optimizer_generator.step()

        if self._optimizer_mixer is None and self.optimizer_generator is None:
            warnings.warn('Warning. Neither Mixer nor Generator optimizer "\
                    "are set, call to update returned with no effect')

    def sample_generator_input(self, particles=None):
        """
        Args:
            particles (Int): number of sampling particles
        """

        if particles is None:
            particles = self._particles
        
        n_layers = len(self._layer_fns)
        if self._use_mixer:
            assert self._mixer is not None, "option `use_mixer` was set to " \
                    "{}, but mixer is {}. Set mixer before calling sampler " \
                    "method".format(self._use_mixer, self._mixer)
            mixer_input = torch.randn(particles, self._s_dim).to(self._device)
            mixer_output = self._mixer(mixer_input)  # [particles, z_dim*n_layers]
            noise_vec = mixer_output.view(-1, n_layers, self._z_dim)
            noise_vec = noise_vec.transpose(0, 1)  # [n_layers, particles, z_dim]
        else:
            noise_vec = torch.randn(particles, self._z_dim).to(self._device)
            noise_vec = noise_vec.unsqueeze(0).repeat(n_layers, 1, 1)

        return noise_vec

    def sample_parameters(self, noise_vec, particles=None):
        """
        Args:
            noise_vec (torch.tensor): Generator inputs
            particles (Int): number of sampling particles
        """
        if particles is None: 
            particles = self._particles
        params = torch.zeros(particles, self._n_params)
        start = 0
        for noise, layer, dims in zip(noise_vec, self._layer_fns, self._target_dim_map):
            layer_output = layer(noise)
            assert layer_output.shape[1] == dims[0]+dims[1], "wrong shape out. " \
                    "Expected output of shape [{}, {}], but got {}".format(
                            particles,
                            dims[0]+dims[1],
                            layer_output.shape)
            params[:, start:start+dims[0]+dims[1]] = layer_output
            start += dims[0]+dims[1]

        assert start == self._n_params, "Parameter generation ended before "\
                "all parameters were generated. Total parameters generated: "\
                "{} out of total: {}".format(
                        start,
                        self._n_params)

        return params


    def set_parameters_to_model(self, parameters):
        """ Setter. Transfers the generated parameters to the network layers
            for evaluation
            Sets:
        """
        assert (parameters.shape[0] == self._particles and 
                parameters.shape[1] == self._n_params), "Length of tensor "\
                "does not match the expected input size. Expected input "\
                "parameter tensor of size [{}, {}], but got tensor of "\
                "shape {}".format(
                    self._particles,
                    self._n_params,
                    parameters.shape)

        start = 0
        for network_fn, dims in zip(self._network_fns, self._target_dim_map):
            w_expected = dims[0]
            b_expected = dims[1]
            layer_str = dims[2]
            layer_weight = parameters[:, start:start+w_expected]
            start += w_expected
            layer_bias = parameters[:, start:start+b_expected]
            start += b_expected
            network_fn.set_weights(layer_weight)
            network_fn.set_bias(layer_bias)

        
    def forward_model(self, x, batch_size=None, training=True):
        act_fn = self._target_arch_def.activation
        flatten_after = self._target_arch_def.flatten
        for idx, layer in enumerate(self._network_fns):
            x = layer(x)
            x = act_fn(x)
            if idx+1 in flatten_after:
                x = x.view(x.size(0), self._particles, -1)
        return x
       
    def eval(self):
        self._mixer.eval()
        for layer in self._layer_fns:
            layer.eval()

    def train(self):
        self._mixer.train()
        for layer in self._layer_fns:
            layer.train()
