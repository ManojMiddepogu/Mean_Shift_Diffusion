from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)

class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class Guidance_Model(nn.Module):
    def __init__(
        self,
        image_size,
        model_channels,
        out_channels,
        dims=2,
        dropout=0,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        use_scale_shift_norm=False,
    ):
        super().__init__()
        self.image_size = image_size
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.dims = dims
        self.dropout = dropout
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.use_fp16 = use_fp16
        self.dtype = th.float16 if use_fp16 else th.float32
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        reshape_embed_dim = model_channels * (image_size ** 2)
        self.reshape_block = nn.Sequential(
            # CHECK - SHOULD WE UPDATE THE NORMALIZATION FROM 32 to 16??
            normalization(time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, reshape_embed_dim),
        )
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)
        
        self.proj = nn.Sequential(
            normalization(self.model_channels),
            nn.SiLU(),
            zero_module(conv_nd(dims, self.model_channels, self.out_channels, 3, padding = 1)),
        )
        self.sigma_diff = nn.Sequential(
            normalization(self.model_channels),
            nn.SiLU(),
            zero_module(conv_nd(dims, self.model_channels, 1, image_size, padding = 0)),
        )

    def forward(self, timesteps, sqrt_one_minus_alphas_cumprod, y=None):
    # def forward(self, timesteps, y=None):
        """
        :param timesteps: a 1-D [N] batch of timesteps.
        :param y: an [N] Tensor of labels.
        :return: an [N x C x H x W] Tensor of outputs.
        """
        assert (y is not None) , "must specify y"
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels)) #(N,TE)
        label = self.label_emb(y) #(N,TE)
        final_emb = emb + label
        reshape_emb = self.reshape_block(final_emb)
        reshape_emb_image = reshape_emb.view(-1, self.model_channels, self.image_size, self.image_size)
        mu = self.proj(reshape_emb_image)
        sigma_diff = self.sigma_diff(reshape_emb_image).squeeze()
        sigma = _extract_into_tensor(sqrt_one_minus_alphas_cumprod, timesteps, sigma_diff.shape) + F.relu(- sigma_diff)

        # HANDLING THE BASE CASE HERE ITSELF
        t_is_less_zero = (timesteps < 0)
        if t_is_less_zero.any():
            # Creating zero tensors for mu and sigma where timesteps is 0
            zero_mu = th.zeros_like(mu)
            zero_sigma = th.zeros_like(sigma)

            # Replacing the corresponding elements with zeros
            mu[t_is_less_zero] = zero_mu[t_is_less_zero]
            sigma[t_is_less_zero] = zero_sigma[t_is_less_zero]

        return mu, sigma

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
