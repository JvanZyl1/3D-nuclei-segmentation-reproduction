import torch 
from torch import nn 
from dataclasses import dataclass 
from typing import Any
import chainer.links as L

@dataclass 
class Config:
    ''' Pythonic substitute for the repetitive kwargs provided to each model init '''

    ndim: int = 3
    n_class: int = 2
    init_channel: int = 2
    kernel_size: int = 3
    pool_size: int = 2
    ap_factor: int = 2
    class_weight: torch.Tensor = torch.tensor([1,1], dtype=torch.float)
    loss_func: nn.Module = nn.functional.cross_entropy


class Model_L2(nn.Module):
    ''' Translated from `src.lib.model.Model_L2` in the original codebase. '''

    def __init__(self, c=Config()):

        super().__init__()

        self.config = c 
        # this would allow us to move it over to the GPU, along with the rest of the model, ... I think
        self.class_weight = nn.Parameter(c.class_weight, requires_grad=False)

        # chainer.links.ConvolutionND takes the following input args:
        ndim: Any
        in_channels: Any
        out_channels: Any
        ksize: Any | None
        stride: int
        pad: int 
        nobias: bool = False 
        initialW: Any | None = None
        initial_bias: Any | None = None
        cover_all: bool = False
        dilate: int = 1
        groups: int = 1

        # TODO: use HeNormal initialisation from pytorch


        if not c.ndim == 3: 
            # NOTE: pytorch does not support ConvND by default. It's possible
            # to implement it, see https://github.com/pvjosue/pytorch_convNd
            # while it's just one file, the config seems to set D=3 anyway. 

            raise ValueError('Only 3D convolutions are supported')

        # We don't need to provide an explicit initialiser, as the pytorch default is Kaiming (HeNormal)
        # TODO: However, I don't see a clear way to init biases to zero. 
        self.conv0 = nn.Conv3d(1, c.init_channel, c.kernel_size, 1, int(c.kernel_size/2))
        # c0=L.ConvolutionND(ndim, 1, init_channel, kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),
        # c1=L.ConvolutionND(ndim, init_channel, int(init_channel * (ap_factor ** 1)), kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),

        # c2=L.ConvolutionND(ndim, int(init_channel * (ap_factor ** 1)), int(init_channel * (ap_factor ** 1)), kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),
        # c3=L.ConvolutionND(ndim, int(init_channel * (ap_factor ** 1)), int(init_channel * (ap_factor ** 2)), kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),

        # c4=L.ConvolutionND(ndim, int(init_channel * (ap_factor ** 2)), int(init_channel * (ap_factor ** 2)), kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),
        # c5=L.ConvolutionND(ndim, int(init_channel * (ap_factor ** 2)), int(init_channel * (ap_factor ** 3)), kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),

        # dc0=L.DeconvolutionND(ndim, int(init_channel * (ap_factor ** 3)), int(init_channel * (ap_factor ** 3)), self.pool_size, self.pool_size, 0, initialW=initializer, initial_bias=None),
        # dc1=L.ConvolutionND(ndim, int(init_channel * (ap_factor ** 2) + init_channel * (ap_factor ** 3)), int(init_channel * (ap_factor ** 2)), kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),
        # dc2=L.ConvolutionND(ndim, int(init_channel * (ap_factor ** 2)), int(init_channel * (ap_factor ** 2)), kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),

        # dc3=L.DeconvolutionND(ndim, int(init_channel * (ap_factor ** 2)), int(init_channel * (ap_factor ** 2)), self.pool_size, self.pool_size, 0, initialW=initializer, initial_bias=None),
        # dc4=L.ConvolutionND(ndim, int(init_channel * (ap_factor ** 1) + init_channel * (ap_factor ** 2)), int(init_channel * (ap_factor ** 1)), kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),
        # dc5=L.ConvolutionND(ndim, int(init_channel * (ap_factor ** 1)), int(init_channel * (ap_factor ** 1)), kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),

        # dc6=L.ConvolutionND(ndim, int(init_channel * (ap_factor ** 1)), n_class, 1, 1, initialW=initializer, initial_bias=None),

        # set all Conv3d layers biases' to zero 
        for layer in self.children():
            if isinstance(layer, nn.Module):
                nn.init.zeros_(layer.bias)

        # bnc0=L.BatchNormalization(init_channel),
        # bnc1=L.BatchNormalization(int(init_channel * (ap_factor ** 1))),

        # bnc2=L.BatchNormalization(int(init_channel * (ap_factor ** 1))),
        # bnc3=L.BatchNormalization(int(init_channel * (ap_factor ** 2))),

        # bnc4=L.BatchNormalization(int(init_channel * (ap_factor ** 2))),
        # bnc5=L.BatchNormalization(int(init_channel * (ap_factor ** 3))),

        # bndc1=L.BatchNormalization(int(init_channel * (ap_factor ** 2))),
        # bndc2=L.BatchNormalization(int(init_channel * (ap_factor ** 2))),
        # bndc4=L.BatchNormalization(int(init_channel * (ap_factor ** 1))),
        # bndc5=L.BatchNormalization(int(init_channel * (ap_factor ** 1)))

        