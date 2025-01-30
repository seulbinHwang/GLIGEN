import torch
import torch.nn as nn
#import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager

# from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

from ldm.util import instantiate_from_config


class AutoencoderKL(nn.Module):

    def __init__(self, ddconfig, embed_dim, scale_factor=1):
        """
'params': {'scale_factor': 0.18215, 'embed_dim': 4,
'ddconfig':
    {'double_z': True, 'z_channels': 4, 'resolution': 256,
    'in_channels': 3, 'out_ch': 3, 'ch': 128, 'ch_mult': [1, 2, 4, 4],
    'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}}}
        """
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"],
                                          2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim,
                                               ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        self.scale_factor = scale_factor # 0.18215

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior.sample() * self.scale_factor

    def decode(self, z):
        """
        z: (B, C, H, W) = (B, 4, 64, 64)
        """
        z = 1. / self.scale_factor * z
        z = self.post_quant_conv(z) # (B, C, H, W) -> (B, z_channels, H, W)
        dec = self.decoder(z) #
        # dec: (B, 3, H, W)
        return dec
