from enum import Enum, auto
from typing import Tuple, Type

import math
import torch
from omegaconf import DictConfig
from torch import nn

from .layers import LayerNorm
from .layers import  _assert


class PartitionType(Enum):
    WINDOW = auto()
    GRID = auto()
    GLOBAL = auto()


def nChw_2_nhwC(x: torch.Tensor):
    """N C H W -> N H W C
    """
    assert x.ndim == 4
    return x.permute(0, 2, 3, 1).contiguous()


def nhwC_2_nChw(x: torch.Tensor):
    """N H W C -> N C H W
    """
    assert x.ndim == 4
    return x.permute(0, 3, 1, 2).contiguous()


class DownsampleBase(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def output_is_normed():
        raise NotImplementedError

def get_downsample_layer_Cf2Cl(dim_in: int,
                               dim_out: int,
                               downsample_factor: int,
                               downsample_cfg: DictConfig) -> DownsampleBase:
    type = downsample_cfg.type
    if type == 'patch':
        return ConvDownsampling_Cf2Cl(dim_in=dim_in,
                                      dim_out=dim_out,
                                      downsample_factor=downsample_factor,
                                      downsample_cfg=downsample_cfg)
    raise NotImplementedError


class ConvDownsampling_Cf2Cl(DownsampleBase):
    """Downsample with input in NCHW [channel-first] format.
    Output in NHWC [channel-last] format.
    """
    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 downsample_factor: int,
                 downsample_cfg: DictConfig):
        super().__init__()
        assert isinstance(dim_out, int)
        assert isinstance(dim_in, int)
        assert downsample_factor in (2, 4, 8)

        norm_affine = downsample_cfg.get('norm_affine', True)
        overlap = downsample_cfg.get('overlap', True)

        if overlap:
            kernel_size = (downsample_factor - 1)*2 + 1
            padding = kernel_size//2
        else:
            kernel_size = downsample_factor
            padding = 0
        self.conv = nn.Conv2d(in_channels=dim_in,
                              out_channels=dim_out,
                              kernel_size=kernel_size,
                              padding=padding,
                              stride=downsample_factor,
                              bias=False,
                              padding_mode='replicate')

        self.norm = LayerNorm(num_channels=dim_out, eps=1e-5, affine=norm_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = nChw_2_nhwC(x)
        x = self.norm(x)
        return x

    @staticmethod
    def output_is_normed():
        return True


def get_downsample_layer_Cf2Cl(dim_in: int,
                               dim_out: int,
                               downsample_factor: int,
                               downsample_cfg: DictConfig) -> DownsampleBase:
    type = downsample_cfg.type
    if type == 'patch':
        return ConvDownsampling_Cf2Cl(dim_in=dim_in,
                                      dim_out=dim_out,
                                      downsample_factor=downsample_factor,
                                      downsample_cfg=downsample_cfg)
    raise NotImplementedError


class GLU(nn.Module):
    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 channel_last: bool,
                 act_layer: Type[nn.Module],
                 bias: bool = True):
        super().__init__()
        # Different activation functions / versions of the gated linear unit:
        # - ReGLU:  Relu
        # - SwiGLU: Swish/SiLU
        # - GeGLU:  GELU
        # - GLU:    Sigmoid
        # seem to be the most promising once.
        # Extensive quantitative eval in table 1: https://arxiv.org/abs/2102.11972
        # Section 2 for explanation and implementation details: https://arxiv.org/abs/2002.05202
        # NOTE: Pytorch has a native GLU implementation: https://pytorch.org/docs/stable/generated/torch.nn.GLU.html?highlight=glu#torch.nn.GLU
        proj_out_dim = dim_out*2
        self.proj = nn.Linear(dim_in, proj_out_dim, bias=bias) if channel_last else \
            nn.Conv2d(dim_in, proj_out_dim, kernel_size=1, stride=1, bias=bias)
        self.channel_dim = -1 if channel_last else 1

        self.act_layer = act_layer()

    def forward(self, x: torch.Tensor):
        x, gate = torch.tensor_split(self.proj(x), 2, dim=self.channel_dim)
        return x * self.act_layer(gate)
    
    
class MLP(nn.Module):
    def __init__(self,
                 dim: int,
                 channel_last: bool,
                 expansion_ratio: int,
                 act_layer: Type[nn.Module],
                 gated: bool = True,
                 bias: bool = True,
                 drop_prob: float = 0.):
        super().__init__()
        inner_dim = int(dim * expansion_ratio)
        if gated:
            # To keep the number of parameters (approx) constant regardless of whether glu == True
            # Section 2 for explanation: https://arxiv.org/abs/2002.05202
            #inner_dim = round(inner_dim * 2 / 3)
            #inner_dim = math.ceil(inner_dim * 2 / 3 / 32) * 32 # multiple of 32
            #inner_dim = round(inner_dim * 2 / 3 / 32) * 32 # multiple of 32
            inner_dim = math.floor(inner_dim * 2 / 3 / 32) * 32 # multiple of 32
            proj_in = GLU(dim_in=dim, dim_out=inner_dim, channel_last=channel_last, act_layer=act_layer, bias=bias)
        else:
            proj_in = nn.Sequential(
                nn.Linear(in_features=dim, out_features=inner_dim, bias=bias) if channel_last else \
                    nn.Conv2d(in_channels=dim, out_channels=inner_dim, kernel_size=1, stride=1, bias=bias),
                act_layer(),
            )
        self.net = nn.Sequential(
            proj_in,
            nn.Dropout(p=drop_prob),
            nn.Linear(in_features=inner_dim, out_features=dim, bias=bias) if channel_last else \
                nn.Conv2d(in_channels=inner_dim, out_channels=dim, kernel_size=1, stride=1, bias=bias)
        )


    def forward(self, x):
        x = self.net(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float=1e-5, inplace: bool=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        gamma = self.gamma
        return x.mul_(gamma) if self.inplace else x * gamma
    
\
def window_partition(x: torch.Tensor, window_size: Tuple[int, int]) -> torch.Tensor:
    B, H, W, C = x.shape
    _assert(H % window_size[0] == 0, f'height ({H}) must be divisible by window ({window_size[0]})')
    _assert(W % window_size[1] == 0, f'width ({W}) must be divisible by window ({window_size[1]})')
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(windows, window_size: Tuple[int, int], img_size: Tuple[int, int]) -> torch.Tensor:
    H, W = img_size
    C = windows.shape[-1]
    x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    return x


def grid_partition(x, grid_size: Tuple[int, int]) -> torch.Tensor:
    B, H, W, C = x.shape
    _assert(H % grid_size[0] == 0, f'height {H} must be divisible by grid {grid_size[0]}')
    _assert(W % grid_size[1] == 0, f'width {W} must be divisible by grid {grid_size[1]}')
    x = x.view(B, grid_size[0], H // grid_size[0], grid_size[1], W // grid_size[1], C)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, grid_size[0], grid_size[1], C)
    return windows


def grid_reverse(windows, grid_size: Tuple[int, int], img_size: Tuple[int, int]) -> torch.Tensor:
    H, W = img_size
    C = windows.shape[-1]
    x = windows.view(-1, H // grid_size[0], W // grid_size[1], grid_size[0], grid_size[1], C)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, H, W, C)
    return x







