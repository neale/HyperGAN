# Neale Ratzlaff

""" Stein gradient estimator operations """
import torch
import torch.nn.functional as F
import numpy as np


class GradientEstimatorStein(object):
    """ operations to compute gradients w.r.t the stein equations
    The operations that we define are necessary for the following methods:
    1. Amortized Stein Variational Gradient Descent (Feng, Liu. 2017)
        https://arxiv.org/abs/1707.06626
    2. Gradient Estimators for Implicit Models (Li, Turner. 2018)
        https://arxiv.org/pdf/1705.07107.pdf
    """

    def __init__(
            self, 
            particles,
            device,
            data_loss_fn=F.cross_entropy,
            loss_grad_reduction=torch.sum,
            kernel_sharpness=2.0,
            alpha=1.0,
            nu=1e-3,
            estimator='svgd',
            gradient_info=None,
            name="GradientEstimatorStein"):
        """
        Args:
            Args for computing gradients w.r.t the stein identity 
            ====================================================================
            particles (int): the number of particles sampled from the 
                hypernetwork at each iteration
            device (torch.device): cannot be None, device for tensor creation
            data_loss_fn (Function: torch.nn.functional): loss function used to
                update the sampled particles in the direction of the likelihod
            loss_grad_reduction (Function: torch): functional that returns a 
                scalar from a tensor e.g. torch.sum, torch.mean
            kernel sharpness (float): used to calculate the kernel bandwidth
                in the SVGD update
            alpha (float): controls the tradeoff between the likelihood term
                and the repulsive term in amortized SVGD
            nu (float): regularization for inverse kernel (stein gradient
                estimator)
            estimator (str): either amortized svgd (svgd) or stein gradient 
                estimator (stein)
            gradient_info (dict): used keys are "inputs" for grad variables
                and "grad" for gradients w.r.t those inputs, can be None. 
                useful for bypassing the grad computation if applying gradients
                is the only desired operation
            name (str): 
        """
        
        self._particles = particles
        self._device = device
        self._data_loss_fn = data_loss_fn
        self._loss_grad_reduction = loss_grad_reduction
        self._kernel_sharpness = kernel_sharpness
        self._alpha = alpha
        self._nu = nu
        self._estimator = estimator
        self._gradient_info = gradient_info

    def _grad_likelihood(self, outputs, targets, params=None):
        """ Computes the data loss and the gradient of the likelihood
        Args:
            outputs (torch.tensor [particles, batch_size, output_dim])
                predictions of the hypernetwork samples on the data
            targets (torch.tensor [batch_size, output_dim])
                class labels of the data
            params (torch.tensor) [particles, n_params]
                output parameters of the hypernetwork (optional but required)
                for stein gradient estimator
        Returns:
            loss tensor
            gradient of the loss w.r.t the predictions
        """
        # TODO take gradients w.r.t the generated parameters
        loss = torch.stack([self._data_loss_fn(
            x,
            targets,
            reduction='none') for x in outputs])
        
        output = params if params is not None else outputs
        loss_grad = torch.autograd.grad(
            self._loss_grad_reduction(loss),
            inputs=output)[0]

        return loss, loss_grad
    
    def _rbf_kernel(self, x, y, h_min=1e-3):
        """ computes an rbf kernel between tensors x, y
        Args: 
            x (torch.tensor [Nx, B, D]) containing Nx particles
            y (torch.tensor [Ny, B, D]) containing Ny particles
            h_min(`float`): Minimum bandwidth.
        """
        Nx, Bx, Dx = x.shape 
        Ny, By, Dy = y.shape
        assert (Bx == By and Dx == Dy)
        diff = x.unsqueeze(1) - y.unsqueeze(0) # Nx x Ny x B x D
        dist_sq = torch.sum(diff**2, -1).mean(dim=-1) # Nx x Ny
        values, _ = torch.topk(dist_sq.view(-1), k=dist_sq.nelement()//2+1)
        median_sq = values[-1]

        h = median_sq / np.log(Nx)
        h = torch.max(h, torch.tensor([h_min]).to(self._device))

        # kappa [p//2, p//2], kappa_grad [p/2, p/2, batch, dim_out]
        kappa = torch.exp(-dist_sq / h)
        kappa_grad = torch.einsum('ij,ijkl->ijkl', kappa, -2 * diff / h)
        return kappa, kappa_grad

    def _score_func(self, x, h_min=1e-3):
        """ Computes -(K(x_j, x_i) + \nu I)^-1
            following the stein gradient estimator
        Args: 
            x (torch.tensor) [N, D]) containing N particles
            h_min(`float`): Minimum bandwidth.
        """
        N, D = x.shape
        z_x = torch.rand_like(x) * 1e-10
        x += z_x
        diff = x.unsqueeze(1) - x.unsqueeze(0) # N x N x D
        dist_sq = torch.sum(diff**2, -1) # N x N
        values, _ = torch.topk(dist_sq.view(-1), k=dist_sq.nelement()//2+1)
        median_sq = values[-1]
        h = median_sq / np.log(N)
        h = torch.max(h, torch.tensor([h_min]).to(self._device))
        kappa = torch.exp(-dist_sq / h)
        
        I = torch.eye(N).to(self._device)
        kappa_inv = torch.inverse(kappa + self._nu * I)
        kappa_grad = torch.einsum('ij,ijk->jk', kappa, -2 * diff / h)

        return kappa_inv @ kappa_grad
        
    def _svgd_k(self, x, y):
        """ computes K(x_j, x_i) for amortized svgd
        Args: 
            x (torch.tensor [Nx, B, D]) containing Nx particles
            y (torch.tensor [Ny, B, D]) containing Ny particles
            h_min(`float`): Minimum bandwidth.
        """
        kappa, kappa_grad = self._rbf_kernel(x, y)
        return kappa, kappa_grad
    
    def compute_gradients(self, outputs, params, targets):
        """ computes the gradient corresponding to the selected stein form
        Args:
            outputs (torch.tensor [particles, batch_size, output_dim])
                predictions of the hypernetwork samples on the data
            params (torch.tensor [particles, n_parameters])
                parameters generaed by the hypernetwork
            targets (torch.tensor [batch_size, output_dim])
                class labels of the data
        Returns:
            gradient of either amortized SVGD or stein gradient estimator 
        """
        # TODO remove splitting of particles, actually resample 
        
        if self._estimator == 'svgd':
            out_i, out_j = torch.split(outputs, len(outputs)//2, dim=0)
            loss, loss_grad = self._grad_likelihood(out_j, targets)
            eps_i = torch.rand_like(out_i) * 1e-10
            eps_j = torch.rand_like(out_j) * 1e-10
            kappa, kappa_grad = self._svgd_k(out_j+eps_j, out_i+eps_i)
            p_ref = kappa.shape[0]
            logp_grad = torch.einsum('ij, ikl->jkl', kappa, loss_grad) / p_ref
            grad_out = logp_grad + self._alpha * kappa_grad.mean(0)
            outputs = out_i

        elif self._estimator == 'stein':
            loss, loss_grad = self._grad_likelihood(outputs, targets, params)
            logq_grad = self._score_func(params)
            grad_out = loss_grad + logq_grad
            outputs = params

        self.set_gradient_info(outputs, grad_out)

        return loss

    def set_gradient_info(self, inputs, grad):
        """ publicly accessible dictionary of computed gradients
        Args:
            inputs (torch.tensor): the variables that the gradients are
                taken w.r.t
            grad (torch.tensor): a tensor of equal shape to `inputs`, 
                the gradients of some output w.r.t `inputs`
        """

        self._gradient_info = {
                'inputs': inputs, 
                'grad': grad
                }

    def apply_gradients(self):
        if self._gradient_info is None or \
            'inputs' not in self._gradient_info or \
            'grad' not in self._gradient_info:
            
            raise AttributeError("gradient info dictionary malformed, "\
                    "pass in inputs and grads to `set_gradient_info`, else "\
                    "call compute gradients to set internally")
        inputs = self._gradient_info['inputs']
        grads = self._gradient_info['grad']
        torch.autograd.backward(inputs, grad_tensors=grads.detach())
        
