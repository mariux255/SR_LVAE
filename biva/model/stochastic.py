from typing import *

import torch
from torch import nn, Tensor
from torch.distributions import Normal

from ..layers import PaddedNormedConv, NormedDense
from ..utils import batch_reduce

# test
import torchvision
import math
import torch
from torch import nn, Tensor
from torch.nn.functional import softplus
from torch.distributions import Distribution
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from functools import reduce


class StochasticLayer(nn.Module):
    """
    An abstract class of a VAE stochastic layer.
    """

    def __init__(self, data: Dict, tensor_shp: Tuple[int], **kwargs: Any):
        super().__init__()
        self._output_shape = None
        self._input_shape = tensor_shp

    def forward(self, x: Optional[Tensor], inference: bool, sample: bool = True, N: Optional[int] = None, **kwargs) -> \
            Tuple[
                Tensor, Dict[str, Any]]:
        """
        Returns the distribution parametrized by x and sample if `sample1`=True. If no hidden state is provided, uses the prior.
        :param x: hidden state used to computed logits (Optional : None means using the prior)
        :param inference: inference mode
        :param sample: sample layer
        :param N: number of samples (when sampling from prior)
        :param kwargs: additional args passed ot the stochastic layer
        :return: (projected sample, data)
        """
        raise NotImplementedError

    def loss(self, q_data: Dict[str, Any], p_data: Dict[str, Any], **kwargs: Any) -> Dict[str, List[Any]]:
        """
        Compute the KL divergence and other auxiliary losses if required
        :param q_data: data received from the posterior forward pass
        :param p_data: data received from the prior forward pass
        :param kwargs: other parameters passed to the kl function
        :return: dictionary of losses {'kl': [values], 'auxiliary' : [aux_values], ...}
        """
        raise NotImplementedError

    @property
    def output_shape(self):
        return self._output_shape

    @property
    def input_shape(self):
        return self._input_shape


class DenseNormal(StochasticLayer):
    """
    A Normal stochastic layer parametrized by dense layers.
    """

    def __init__(self, data: Dict, tensor_shp: Tuple[int], top: bool = False, act: nn.Module = nn.ELU,
                 weightnorm: bool = True, log_var_act: Optional[Callable] = nn.Softplus, **kwargs):
        super().__init__(data, tensor_shp)

        self._input_shape = tensor_shp

        self.eps = 1e-8
        self.nz = data.get('N')
        self.tensor_shp = tensor_shp
        self.dim = 2
        self.act = act()
        self.log_var_act = log_var_act() if log_var_act is not None else None

        # stochastic layer and prior
        if top:
            prior = torch.zeros((2 * self.nz))
            #print("PRIOR SHAPE: ", prior.shape)
            self.register_buffer('prior', prior)

        # computes logits
        nz_in = 2 * self.nz
        self.qx2z = NormedDense(tensor_shp, nz_in, weightnorm=weightnorm)
        if not top:
            self.px2z = NormedDense(tensor_shp, nz_in, weightnorm=weightnorm)

        self._output_shape = (-1, self.nz)

    @property
    def output_shape(self):
        return self._output_shape

    @property
    def input_shape(self):
        return self._input_shape

    def compute_logits(self, x: Tensor, inference: bool) -> Tuple[Tensor, Tensor]:
        """
        Compute the logits of the distribution.
        :param x: input tensor
        :param inference: inference mode
        :return: logits
        """
        x = self.act(x)

        if inference:
            logits = self.qx2z(x)
        else:
            logits = self.px2z(x)

        # apply activation to logvar
        mu, logvar = logits.chunk(2, dim=1)
        if self.log_var_act is not None:
            logvar = self.log_var_act(logvar)
        return mu, logvar

    def forward(self, x: Optional[Tensor], inference: bool, sample: bool = True, N: Optional[int] = None, **kwargs) -> \
            Tuple[Tensor, Dict[str, Any]]:

        if x is None:
            mu, logvar = self.prior.expand(N, *self.prior.shape).chunk(2, dim=1)
        else:
            mu, logvar = self.compute_logits(x, inference)

        # sample layer
        std = logvar.mul(0.5).exp()
        dist = Normal(mu, std)

        z = dist.rsample() if sample else None

        return z, {'z': z, 'dist': dist}

    def loss(self, q_data: Dict[str, Any], p_data: Dict[str, Any], **kwargs: Any) -> Dict[str, List]:
        z_q = q_data.get('z')
        q = q_data.get('dist')
        p = p_data.get('dist')

        kl = q.log_prob(z_q) - p.log_prob(z_q)
        kl = batch_reduce(kl)

        return {'kl': [kl]}


class ConvNormal(StochasticLayer):
    """
    A Normal stochastic layer parametrized by convolutions.
    """

    def __init__(self, data: Dict, tensor_shp: Tuple[int], top: bool = False, act: nn.Module = nn.ELU,
                 learn_prior: bool = False, weightnorm: bool = True, log_var_act: Optional[Callable] = nn.Softplus,
                 **kwargs):
        super().__init__(data, tensor_shp)

        self.eps = 1e-8
        nhid = tensor_shp[1]
        self.nz = data.get('N')
        kernel_size = data.get('kernel')
        self.tensor_shp = tensor_shp
        self.input_shp = tensor_shp
        self.act = act()
        self.log_var_act = log_var_act() if log_var_act is not None else None
        self.top = top
        # prior
        if top:
            prior = torch.zeros((2 * self.nz, *tensor_shp[2:]))
            #print("PRIOR SHAPE: ", prior)
            if learn_prior:
                self.prior = nn.Parameter(prior)
            else:
                self.register_buffer('prior', prior)

        # computes logits
        nz_in = 2 * self.nz
        self.qx2z = PaddedNormedConv(tensor_shp, nn.Conv2d(nhid, nz_in, kernel_size), weightnorm=weightnorm)
        if not top:
            self.px2z = PaddedNormedConv(tensor_shp, nn.Conv2d(nhid, nz_in, kernel_size), weightnorm=weightnorm)

        # compute output shape
        nz_out = self.nz
        out_shp = (-1, nz_out, *tensor_shp[2:])
        self._output_shape = out_shp
        self._input_shape = tensor_shp

        self.downsized = nn.Sequential( # Take big image, return small image
            nn.Conv2d(3, 5, 3, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(5, 7, 3, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(7, 8, 3),
            nn.Flatten(start_dim = 1, end_dim = -1),
            nn.Linear(in_features=800, out_features=500),
            nn.LeakyReLU(negative_slope=0.01),
            #nn.Linear(in_features=500, out_features=150),
            #nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(in_features=500, out_features=20*4*4),
            #nn.Unflatten(dim = 1, unflattened_size=(20,4,4)),
        )

    @property
    def output_shape(self):
        return self._output_shape

    @property
    def input_shape(self):
        return self._input_shape

    def compute_logits(self, x: Tensor, inference: bool) -> Tuple[Tensor, Tensor]:
        """
        Compute the logits of the distribution.
        :param x: input tensor
        :param inference: inference mode
        :return: logits
        """
        x = self.act(x)
        if inference:
            logits = self.qx2z(x)
        else:
            logits = self.px2z(x)

        # apply activation to logvar
        mu, logvar = logits.chunk(2, dim=1)
        if self.log_var_act is not None:
            logvar = self.log_var_act(logvar)
        return mu, logvar

    def expand_prior(self, batch_size: int,image = None):
        #print("prior shape = ", self.prior.shape)
        #print("Batch size = ", batch_size)
        if type(image) != type(None):
            N = image.shape
            N = N[0]
            h_y = self.downsized(image)
            h_y = h_y.view(-1, *(20,4,4))
            return h_y.expand(N, *self.prior.shape).chunk(2,dim=1)
            #return self.prior.expand(batch_size, *self.prior.shape).chunk(2, dim=1)
        return self.prior.expand(batch_size, *self.prior.shape).chunk(2, dim=1)

    def forward(self, x: Optional[Tensor], inference: bool, sample: bool = True, N: Optional[int] = None, **kwargs) -> \
            Tuple[
                Tensor, Dict[str, Any]]:
        #print("TOP: ",inference)
        if x is None:
            mu, logvar = self.expand_prior(N)
            #print("MU: ", mu.shape)
            #print("LOGVAR: ", logvar.shape)
        elif self.top and not inference:
            #print("HIt it")
            mu, logvar = self.expand_prior(N,image=x)
        else:
            mu, logvar = self.compute_logits(x, inference)


        # sample layer
        std = logvar.mul(0.5).exp()
        dist = Normal(mu, std)

        z = dist.rsample() if sample else None

        return z, {'z': z, 'dist': dist}

    def loss(self, q_data: Dict[str, Any], p_data: Dict[str, Any], **kwargs: Any) -> Dict[str, List]:
        z_q = q_data.get('z')
        q = q_data.get('dist')
        p = p_data.get('dist')

        kl = q.log_prob(z_q) - p.log_prob(z_q)
        kl = batch_reduce(kl)

        return {'kl': [kl]}
