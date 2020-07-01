# Copyright (c) 2020 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearLayer(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 activation=torch.relu_,
                 particles=1,
                 use_bias=True,
                 device=torch.device('cpu')):

        super(LinearLayer, self).__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._activation = activation
        self._use_bias = use_bias
        self._particles = particles
        self._device = device

        self._weight_dim = input_dim * output_dim
        self.set_weights(torch.randn(self._particles, self._weight_dim))
        if self._use_bias:
            self._bias_dim = output_dim
            self.set_bias(torch.randn(self._particles, self._bias_dim))
        else:
            self._bias_dim = 0
            self._bias = None
    
    @property
    def weights(self):
        return self._weights

    @property 
    def bias(self):
        return self._bias

    @property
    def weight_dim(self):
        return self._weight_dim

    @property
    def bias_dim(self):
        return self._bias_dim

    def particles(self):
        return self._particles

    def set_weights(self, weights):
        assert (weights.shape[1] == self._weight_dim and weights.ndim == 2), (
                "Input weights are of incorrect shape. Expected tensor of " \
                "size [n, {}], but got {} of shape {}".format(
                    self._weight_dim,
                    type(weights),
                    weights.shape))
        
        if self._particles is None:
            if weights.size(0) == 1:
                self._particles = 1

            elif weights.size(0) > 1:
                self._particles = weights.size(0)
        else:
            assert (self._particles == weights.size(0)), (
                "input particle size does not match the predefined " \
                "particle size of {}, either change particle size or change "\
                "inputs".format(
                    self._particles))


        self._weights = weights.view(
                self._particles,
                self._output_dim,
                self._input_dim)

        self._weights = self._weights.to(self._device)
    
    def set_bias(self, bias):    
        assert (bias.shape[1] == self._bias_dim and bias.ndim == 2), (
                "Input bias is of incorrect shape. Expected tensor of " \
                "size [n, {}], but got {} of shape {}".format(
                            self._bias_dim,
                            type(bias),
                            bias.shape))

        if self._particles is None:
            if bias.size(0) == 1:
                self._particles = 1

            elif bias.size(0) > 1:
                self._particles = bias.size(0)
        else:
            assert (self._particles == bias.size(0)), (
                "input particle size does not match the predefined " \
                "particle size of {}, either change particle size or change "\
                "inputs".format(
                    self._particles))
        
        self._bias = bias
        self._bias = self._bias.to(self._device)

    def forward(self, inputs):
        """ Forward method for the defined Linear layer
        Args:
            inputs (torch.tensor) of shape   [B, D]    for particles=1
                                             [B, n, D] for particles=n
            B: batch size
            n: number of particles
            D: input data dimension
        
        Returns:
            torch.tensor with shape [B, D]    for particles=1
                                    [B, n, D] for particles=n
            B: batch size
            n: number of particles
            D: output data dimension
        """
        if self._particles == 1:
            assert (inputs.ndim == 2 and inputs.shape[1] == self.input_dim), (
                "Input is of incorrect shape. Expected tensor of " \
                "size [batch size, {}]".format(
                    self._input_dim))
            inputs = inputs.unsqueeze(0)
        else:
            if inputs.ndim == 2:
                assert (inputs.shape[1] == self.input_dim), (
                        "Input is of incorrect shape. Expected tensor of " \
                        "size [batch size, {}, {}]".format(
                            self._particles,
                            self._input_dim))
                inputs = inputs.unsqueeze(0).expand(self._particles, *inputs.shape)

            elif inputs.ndim == 3:
                assert(
                    inputs.shape[1] == self._particles
                    and inputs.shape[2] == self._input_dim
                ), ("Input is of incorrect shape. Expected tensor of " \
                    "size [B, {}, {}]".format(
                        inputs.shape,
                        self._particles,
                        self._input_dim))
                inputs = inputs.transpose(0, 1) # [n, B, D]
            else:
                raise ValueError("Incorrect Input Shape: {}".format(inputs.shape))
        
        if self._bias is not None:
            x = torch.baddbmm(
                    self._bias.unsqueeze(1), inputs, self._weights.transpose(1, 2))
        else:
            x = torch.bmm(inputs, self._weights.transpose(1, 2))
        x = x.transpose(0, 1)  # [B, n, D]
        x = x.squeeze()

        return self._activation(x)


class Conv2DLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 activation=torch.relu_,
                 strides=1,
                 pooling_fn=None,
                 pooling_kernel=None,
                 pooling_stride=None,
                 padding=0,
                 particles=1,
                 use_bias=True,
                 device=torch.device('cpu')):
        super(Conv2DLayer, self).__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._activation = activation
        self._kernel_size = (kernel_size, kernel_size)
        self._kH, self._kW = self._kernel_size
        self._strides = strides
        self._pooling_fn = pooling_fn
        self._pooling_kernel = pooling_kernel
        self._pooling_stride = pooling_stride
        self._padding = padding
        self._use_bias = use_bias
        self._particles = particles
        self._device = device

        self._weight_dim = out_channels * in_channels * self._kH * self._kW
        self.set_weights(torch.randn(self._particles, self._weight_dim))
        if use_bias:
            self._bias_dim = out_channels
            self.set_bias(torch.randn(self._particles, self._bias_dim))
        else:
            self._bias_dim = 0
            self._bias = None

    @property
    def weights(self):
        return self._weights

    @property
    def bias(self):
        return self._bias

    @property
    def weight_dim(self):
        return self._weight_dim

    @property
    def bias_dim(self):
        return self._bias_dim

    def set_weights(self, weights):
        assert (weights.ndim == 2 and weights.shape[1] == self._weight_dim), (
                "Input weights are of incorrect shape. Expected tensor of " \
                "shape [n, {}], but got {} of shape {}".format(
                        self._weight_dim,
                        type(weights),
                        weights.shape))
        if weights.shape[0] == 1:
            self._particles = 1
            self._weights = weights.view(self._out_channels, self._in_channels,
                                         self._kH, self._kW)
        else:
            self._particles = weights.shape[0]
            weights = weights.view(
                    self._particles,
                    self._out_channels,
                    self._in_channels,
                    self._kH, self._kW)

            self._weights = weights.reshape(
                    self._particles * self._out_channels,
                    self._in_channels, self._kH, self._kW)
        
        self._weights = self._weights.to(self._device)
            
    def set_bias(self, bias):
        assert (bias.ndim == 2 and bias.shape[1] == self._bias_dim), (
            "Input bias is of incorrect shape. Expected tensor of " \
            "shape [n, {}], but got {} of shape {}".format(
                        self._bias_dim,
                        type(bias),
                        bias.shape))
        
        if self._particles is None:
            if self._particles == 1:
                assert bias.shape[0] == 1, (
                    "Input bias is of incorrect shape {}. Expected tensor of " \
                    "shape [{}, {}]".format(bias.shape, 1, self.bias_dim))
            else:
                # parallel weight
                assert bias.shape[0] == self._particles, (
                    "Input bias is of incorrect shape {}. Expected tensor of " \
                    "shape [{}, {}]".format(
                        bias.shape,
                        self._group,
                        self.bias_dim))

        else:
            assert (self._particles == bias.size(0)), (
                    "Input particle size does not match the predefined " \
                    "particle size of {}, either change particle size or change "\
                    "inputs".format(
                        self._particles))
        self._bias = bias.reshape(-1)
        self._bias = self.bias.to(self._device)


    def forward(self, img, keep_group_dim=True):
        """ Forward method for the defined Convolutional layer
        Args:
            img (torch.Tensor): with shape ``[B, C, H, W]    (particles=1)`` 
                                        or ``[B, n, C, H, W] (particles=n)``

                - B: batch size
                - n: number of replicas
                - C: number of channels
                - H: image height
                - W: image width.
                When the shape of img is ``[B, C, H, W]``, all the n 2D Conv
                operations will take img as the same shared input.
                When the shape of img is ``[B, n, C, H, W]``, each 2D Conv operator
                will have its own input data by slicing img.
        Returns:
            torch.Tensor with shape ``[B, n, C', H', W']``
                where the meaning of the symbols are:
                - B: batch
                - n: number of replicas
                - C': number of output channels
                - H': output height
                - W': output width
        """
        if self._particles == 1:
            # non-parallel layer
            assert (img.ndim == 4 and img.shape[1] == self._in_channels), (
                "Input img has wrong shape {}. Expecting [B, {}, H, W]".format(
                    img.shape,
                    self._in_channels))
        else:
            # parallel layer
            if img.ndim == 4:
                if img.shape[1] == self._in_channels:
                    # case 1: non-parallel input
                    img = img.repeat(1, self._particles, 1, 1)
                else:
                    # case 2: parallel input
                    assert img.shape[1] == self._particles * self._in_channels, (
                        "Input img has wrong shape {}. Expecting (B, {}, H, W) "\
                        "or (B, {}, H, W)".format(
                            img.shape,
                            self._in_channels,
                            self._particles * self._in_channels))

            elif img.ndim == 5:
                # case 3: parallel input with unmerged group dim
                assert (
                    img.shape[1] == self._particles
                    and img.shape[2] == self._in_channels
                ), ("Input img has wrong shape {}. "\
                    "Expecting (B, {}, {}, H, W)".format(
                        img.shape,
                        self._particles,
                        self._in_channels))
                # merge groups and channels
                img = img.reshape(img.shape[0], img.shape[1] * img.shape[2],
                                  *img.shape[3:])
            else:
                raise ValueError("Incorrect Image Shape: {}".format(
                    img.shape))

        res = self._activation(
            F.conv2d(
                img,
                self._weights,
                bias=self._bias,
                stride=self._strides,
                padding=self._padding,
                groups=self._particles))
        if (
            self._pooling_kernel is not None
            and self._pooling_fn is not None
            and self._pooling_stride is not None):

            res = self._pooling_fn(
                    res,
                    kernel_size=self._pooling_kernel,
                    stride=self._pooling_stride)

        if self._particles > 1 and keep_group_dim:
            # reshape back: [B, n*C', H', W'] -> [B, n, C', H', W']
            res = res.reshape(res.shape[0], self._particles, self._out_channels,
                              res.shape[2], res.shape[3])

        return res
        
