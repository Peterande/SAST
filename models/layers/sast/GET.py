"""
Part of this code stems from rwightman's MaxVit implementation:
https://github.com/huggingface/pytorch-image-models/blob/1885bdc4318cc3be459981ea1a26cd862220864d/timm/models/maxxvit.py
that is:
- LayerScale
- PartitionAttentionCl
- window*
- grid*
- SelfAttentionCl
"""

from enum import Enum, auto
from functools import partial
from typing import Optional, Union, Tuple, List, Type

import math
import torch
from omegaconf import DictConfig
from torch import nn
import torch.nn.functional as F

from .layers import DropPath, LayerNorm
from .layers import get_act_layer, get_norm_layer
from .layers import to_2tuple, _assert
from einops import rearrange, repeat
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd.function import InplaceFunction
# from .grid_sample1d import GridSample1d
import random
from torch.multiprocessing import Pool
from PIL import Image
import numpy as np
import os
class PartitionType(Enum):
    WINDOW = auto()
    GRID = auto()
    GLOBAL = auto()
from timm.models.layers import create_conv2d, create_pool2d

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


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float=1e-5, inplace: bool=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        gamma = self.gamma
        return x.mul_(gamma) if self.inplace else x * gamma


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
        return self.net(x)


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
                              bias=False)
        self.norm = LayerNorm(num_channels=dim_out, eps=1e-5, affine=norm_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = nChw_2_nhwC(x)
        x = self.norm(x)
        return x

    @staticmethod
    def output_is_normed():
        return True

class GTA(nn.Module):
    """ Group Token Aggregation Layer.
    Including Overlapping Group Convolution and Max Pooling.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        group_num (int): Number of groups.
    """

    def __init__(self, dim_in, dim_out, downsample_factor, group_num=8, embed_dim=20):
        super().__init__()
        self.dim = dim_in
        mul = dim_in // embed_dim
        self.channelexpand = OGConv(dim_in, dim_out, downsample_factor, norm_layer=nn.LayerNorm, group_num=group_num // mul)
        self.pool = Pool()

    def forward(self, x):

        
        # convpool
        x = self.channelexpand(x)
        x = self.pool(x)
        if self.dim==20:
            x = self.pool(x)
        return x  # (B, H//2*W//2, 2C)
    
    @staticmethod
    def output_is_normed():
        return True    

class Pool(nn.Module):
    def __init__(self, pad_type='same', kernel_size=3):
        super().__init__()
        self.pool = create_pool2d('max', kernel_size=kernel_size, stride=2, padding=pad_type)

    def forward(self, x):
        """
        x is expected to have shape (B, C, H, W)
        """
        x = self.pool(x)
        return x  # (B, C, H//2, W//2)
    
class OGConv(nn.Module):
    def __init__(self, dim_in, dim_out, downsample_factor, kernel=3, norm_layer=None, group_num=None):
        super().__init__()
        
        self.in_chans = kernel * dim_in // group_num
        self.step = (kernel - 1) * dim_in // group_num
        self.new_group_num = group_num // 2
        self.padding = ((1 - 1) + 1 * (3 - 1)) // 2
        self.conv = nn.Conv2d(
            in_channels=self.new_group_num * self.in_chans, out_channels=dim_out,
            kernel_size=(3,3), groups=self.new_group_num, padding=self.padding)
        self.norm = norm_layer(dim_out)

    def forward(self, x):  # (B C H W)
        """
        Overlapping group convolution layer.
        """
        B, C, H, W = x.shape
        pad_c = ((self.new_group_num - 1) * (C // self.new_group_num) + self.in_chans - C)
        x = torch.cat([x[:, :pad_c], x], dim=1)
        x_grouped = x.unfold(1, self.in_chans, self.step).permute(0, 1, 4, 2, 3)
        x_grouped = x_grouped.reshape(B, self.new_group_num * self.in_chans, H, W)
        out = self.conv(x_grouped)
        y = self.norm(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        return y  # (B C H W) 


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, dim_head, window_size=(8, 10), attn_drop=0., proj_drop=0., rpe_hidden_dim=512):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = dim // dim_head

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((self.num_heads, 1, 1))), requires_grad=True)
        # mlp to generate table of relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, rpe_hidden_dim, bias=True),
                                     nn.ReLU(inplace=True),
                                     LinearFP32(rpe_hidden_dim, self.num_heads, bias=False))
        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w]))\
                                .permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
        relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
                    torch.abs(relative_coords_table) + 1.0) / np.log2(8)  # log8
        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(dim))
        self.v_bias = nn.Parameter(torch.zeros(dim))


        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape

        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = custom_normalize(q.float(), dim=-1, eps=5e-5)
        k = custom_normalize(k.float(), dim=-1, eps=5e-5)
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01, device=x.device))).exp()
        attn = (q @ k.transpose(-2, -1)) * logit_scale.float()

        # relative_position_bias_table: 2*Wh-1 * 2*Ww-1, nH
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = attn.type_as(x)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x  # (num_windows*B, N, C)


class GroupAttention(nn.Module):
    """ Group based W-MSA module with relative group bias.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size=(8, 10), attn_drop=0., proj_drop=0., rpe_hidden_dim=512, group_num=8, embed_dim=64):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads_i = window_size[0]
        
        # get relative_coords_table
        self.group = group_num // (dim // embed_dim)  # Current group number.
        self.dpg = dim // self.group  # Dimension per group.
        
        self.cpb_mlp_i = nn.Sequential(nn.Linear(2, rpe_hidden_dim, bias=True),
                                    nn.ReLU(inplace=True),
                                    LinearFP32(rpe_hidden_dim, self.num_heads_i, bias=False))
        relative_coords_dpg = torch.arange(-(self.dpg - 1), self.dpg, dtype=torch.float32)
        relative_coords_group = torch.arange(-(self.group - 1), self.group, dtype=torch.float32)
        relative_coords_table_i = torch.stack(torch.meshgrid([relative_coords_dpg, relative_coords_group])) \
            .permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*dpg-1, 2*group-1, 2
        relative_coords_table_i[:, :, :, 0] = relative_coords_table_i[:, :, :, 0] / (self.dpg - 1) if self.dpg > 1 else 0
        relative_coords_table_i[:, :, :, 1] = relative_coords_table_i[:, :, :, 1] / (self.group - 1) if self.group > 1 else 0
        relative_coords_table_i *= 8  # normalize to -8, 8
        relative_coords_table_i = torch.sign(relative_coords_table_i) * torch.log2(
            torch.abs(relative_coords_table_i) + 1.0) / np.log2(8)  # log8
        self.register_buffer("relative_coords_table_i", relative_coords_table_i)

        # get pair-wise relative position index for each group.
        coords_dpg = torch.arange(self.dpg)
        coords_div = torch.arange(self.group)
        coords_i = torch.stack(torch.meshgrid([coords_dpg, coords_div]))  # 2, dpg, group
        coords_flatten_i = torch.flatten(coords_i, 1)  # 2, C
        relative_coords_i = coords_flatten_i[:, :, None] - coords_flatten_i[:, None, :]  # 2, C, C
        relative_coords_i = relative_coords_i.permute(1, 2, 0).contiguous()  # C, C, 2
        relative_coords_i[:, :, 0] += self.dpg - 1  # shift to start from 0
        relative_coords_i[:, :, 1] += self.group - 1
        relative_coords_i[:, :, 0] *= 2 * self.group - 1
        relative_position_index_i = relative_coords_i.sum(-1)  # C, C
        self.register_buffer("relative_position_index_i", relative_position_index_i)

        self.qkv_i = nn.Linear(self.window_size[0] * self.window_size[1],
                               self.window_size[0] * self.window_size[1] * 3, bias=False)

        self.q_i_bias = nn.Parameter(torch.zeros(self.window_size[0] * self.window_size[1]))
        self.v_i_bias = nn.Parameter(torch.zeros(self.window_size[0] * self.window_size[1]))

        self.attn_drop_i = nn.Dropout(attn_drop)
        self.proj_i = nn.Linear(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1])
        self.proj_drop_i = nn.Dropout(proj_drop)
        self.softmax_i = nn.Softmax(dim=-1)


    def forward(self, x_i):
        """
        Args:
            x_i: input features with shape of (num_windows*B, N, C)
        """
        B_, N, C = x_i.shape

        qkv_i_bias = None
        if self.q_i_bias is not None:
            qkv_i_bias = torch.cat(
                (self.q_i_bias, torch.zeros_like(self.v_i_bias, requires_grad=False), self.v_i_bias))
        qkv_i = F.linear(input=x_i.permute(0, 2, 1), weight=self.qkv_i.weight, bias=qkv_i_bias)
        qkv_i = qkv_i.reshape(B_, C, 3, self.num_heads_i, -1).permute(2, 0, 3, 1, 4)
        # qkv_i = self.qkv_i(x).reshape(B_, C, 3, self.num_heads, N // self.num_heads).permute(2, 0, 3, 1, 4)
        q_i, k_i, v_i = qkv_i[0], qkv_i[1], qkv_i[2]
        q_i = custom_normalize(q_i.float(), dim=-1, eps=5e-5)
        k_i = custom_normalize(k_i.float(), dim=-1, eps=5e-5)
        #logit_scale_i = torch.clamp(self.logit_scale_i, max=torch.log(torch.tensor(1. / 0.01, device=x_i.device))).exp()
        attn_i = (q_i @ k_i.transpose(-2, -1)) #* logit_scale_i.float()
        
        relative_position_bias_table_i = self.cpb_mlp_i(self.relative_coords_table_i).view(-1, self.num_heads_i)
        relative_position_bias_i = relative_position_bias_table_i[self.relative_position_index_i.view(-1)].view(
            C, C, -1)  # C,C,nH
        relative_position_bias_i = relative_position_bias_i.permute(2, 0, 1).contiguous()  # nH, C, C
        relative_position_bias_i = 16 * torch.sigmoid(relative_position_bias_i)
        attn_i = attn_i + relative_position_bias_i.unsqueeze(0)

        attn_i = self.softmax_i(attn_i)
        attn_i = attn_i.type_as(x_i)
        attn_i = self.attn_drop_i(attn_i)
        x_i = (attn_i @ v_i).transpose(1, 2).reshape(B_, C, N)

        x_i = self.proj_i(x_i)
        x_i = self.proj_drop_i(x_i)
        return x_i.transpose(1, 2)  # (num_windows*B, N, C)
    

def custom_normalize(input, p=2, dim=1, eps=1e-12, out=None):
    if out is None:
        denom = input.norm(p, dim, keepdim=True).expand_as(input)
        return input / (denom + eps)
    else:
        denom = input.norm(p, dim, keepdim=True).expand_as(input)
        return torch.div(input, denom + eps, out=out)
    
class LinearFP32(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearFP32, self).__init__(in_features, out_features, bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input.float(), self.weight.float(),
                        self.bias.float() if self.bias is not None else None)
    
class PartitionAttentionCl(nn.Module):
    """ Grid or Block partition + Attn + FFN.
    NxC 'channels last' tensor layout.

    According to RW, NHWC attention is a few percent faster on GPUs (but slower on TPUs)
    https://github.com/rwightman/pytorch-image-models/blob/4f72bae43be26d9764a08d83b88f8bd4ec3dbe43/timm/models/maxxvit.py#L1258
    """

    def __init__(
            self,
            dim: int,
            attention_cfg: DictConfig,
            skip_first_norm: bool=False,
            first_block: bool=False,
    ):
        super().__init__()
        norm_eps = attention_cfg.get('norm_eps', 1e-5)
        partition_size = attention_cfg.partition_size
        use_torch_mha = attention_cfg.use_torch_mha
        dim_head = attention_cfg.get('dim_head', 32)
        attention_bias = attention_cfg.get('attention_bias', True)
        mlp_act_string = attention_cfg.mlp_activation
        mlp_gated = attention_cfg.mlp_gated
        mlp_bias = attention_cfg.get('mlp_bias', True)
        mlp_expand_ratio = attention_cfg.get('mlp_ratio', 4)

        drop_path = attention_cfg.get('drop_path', 0.0)
        drop_mlp = attention_cfg.get('drop_mlp', 0.0)
        ls_init_value = attention_cfg.get('ls_init_value', 1e-5)

        assert isinstance(use_torch_mha, bool)
        assert isinstance(mlp_gated, bool)
        assert_activation_string(activation_string=mlp_act_string)
        mlp_act_layer = get_act_layer(mlp_act_string)
        self.window_pruning_only = False

        self_attn_module = WindowAttention   
        scale_dim = dim if self.window_pruning_only else 2 * dim       

        if isinstance(partition_size, int):
            partition_size = to_2tuple(partition_size)
        else:
            partition_size = tuple(partition_size)
            assert len(partition_size) == 2
        self.partition_size = partition_size

        norm_layer = partial(get_norm_layer('layernorm'), eps=norm_eps)  # NOTE this block is channels-last

        self.norm1 = norm_layer(dim)
        self.ls11 = LayerScale(dim=dim, init_values=ls_init_value) if ls_init_value > 0 else nn.Identity()
        self.drop_path11 = DropPath(drop_prob=drop_path) if drop_path > 0 else nn.Identity()

        self.norm12 = norm_layer(dim)
        self.mlp1 = MLP(dim = dim,
                       channel_last=True,
                       expansion_ratio = mlp_expand_ratio,
                       act_layer = mlp_act_layer,
                       gated = mlp_gated,
                       bias = mlp_bias,
                       drop_prob = drop_mlp)
        self.ls12 = LayerScale(dim=dim, init_values=ls_init_value) if ls_init_value > 0 else nn.Identity()
        self.drop_path12 = DropPath(drop_prob=drop_path) if drop_path > 0 else nn.Identity()
        self.sub_layers1 = [self.ls11, self.drop_path11, 
                      self.norm12, self.mlp1, self.ls12, self.drop_path12]
        
        self.win_attn = self_attn_module(dim,
                                         dim_head=dim_head)

        self.norm2 = norm_layer(dim)
        self.ls21 = LayerScale(dim=dim, init_values=ls_init_value) if ls_init_value > 0 else nn.Identity()
        self.drop_path21 = DropPath(drop_prob=drop_path) if drop_path > 0 else nn.Identity()

        self.norm22 = norm_layer(dim)
        self.mlp2 = MLP(dim = dim,
                       channel_last=True,
                       expansion_ratio = mlp_expand_ratio,
                       act_layer = mlp_act_layer,
                       gated = mlp_gated,
                       bias = mlp_bias,
                       drop_prob = drop_mlp)
        self.ls22 = LayerScale(dim=dim, init_values=ls_init_value) if ls_init_value > 0 else nn.Identity()
        self.drop_path22 = DropPath(drop_prob=drop_path) if drop_path > 0 else nn.Identity()
        self.sub_layers2= [self.ls21, self.drop_path21, 
                      self.norm22, self.mlp2, self.ls22, self.drop_path22]
        self_attn_module = GroupAttention
        self.grid_attn = self_attn_module(dim)

        self.first_block = first_block

        self.B = None  # global batch size
        self.N = None  # global number of windows
        self.dim = dim

    def get_indexes_window(self, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, h, w = self.B, self.N, self.partition_size[0], self.partition_size[1]
        temp = h * w * self.dim / 32
        norm_window = (torch.norm(scores, dim=[2, 3], p=3/4) / temp).softmax(-1)
        index_window = get_score_index_2d21d(norm_window.view(B, N), 1 / N, self.bounce_area, randaug=self.training)
        return index_window
    
    def get_indexes_partition(self, scores: torch.Tensor, index_window: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, h, w = self.B, self.N, self.partition_size[0], self.partition_size[1]
        temp = self.dim / 32
        norm_partition = (torch.norm(scores, dim=[3], p=3/4) / temp).view(B * N, -1)[index_window].softmax(-1)
        index_partition, asy_index_partition, K = get_score_index_topk2d(norm_partition, 1 / (h * w), self.bounce_area, randaug=self.training)
        return index_partition, asy_index_partition, K
        
    def _partition_attn(self, x: torch.Tensor, pos_emb: nn.Module, r: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        self.B = x.shape[0]
        img_size = x.shape[1:3]
        self.N = img_size[0] * img_size[1] // (self.partition_size[0] * self.partition_size[1])

        x = x + pos_emb(x)
        x = window_partition(x, self.partition_size).view(self.B * self.N, -1, self.dim)

        shortcut = x
        x = self.win_attn(x)
        for i, layer in enumerate(self.sub_layers1):
            if i == 1 or i == 5:
                x = shortcut + layer(x)
                shortcut = x
            else:
                x = layer(x)
        shortcut = x
        x = self.grid_attn(x)
        for i, layer in enumerate(self.sub_layers2):
            if i == 1 or i == 5:
                x = shortcut + layer(x)
                shortcut = x
            else:
                x = layer(x)
        
        x = window_reverse(x, self.partition_size, (img_size[0], img_size[1]))
        
            
        

        p_loss = 100
        #p_loss = torch.tensor((len(index1) - len(blocked_index1)) // self.B)
        # if self.dim==64:
        #     N, C = paper.shape[0], paper.shape[-1]
        #     win = paper.view(N, -1, C)[:,:,0].unsqueeze(-1)
        #     win[index_window1] = 125
        #     winsliced = win[index_window1].view(-1, 1).contiguous().clone()
        #     winsliced[asy_index1] = 255
        #     winsliced = winsliced.view(M1, -1, 1)
        #     win[index_window1] = winsliced
        #     win = win.view(N, -1, 1)
        #     win = window_reverse(win, self.partition_size, (img_size[0], img_size[1]))
        #     paper = torch.cat([win, win, win], dim=-1)

        #     output_dir = 'vis'
        #     # original_img = Image.open(os.path.join(output_dir, 'image.jpg'))
        #     img_array = (paper[0]).cpu().numpy().astype(np.uint8)
        #     img = Image.fromarray(img_array)
        #     name = 'tokens' + str(self.dim) + '.jpg'
        #     filename = os.path.join(output_dir, name)
        #     img.save(filename)
        #     # img_array = (paper[0]).cpu().numpy().astype(np.uint8)
        #     # visualized_img = Image.fromarray(img_array)

        #     # # 将可视化图层叠加到原始图片上
        #     # visualized_img_resized = visualized_img.resize(original_img.size, Image.ANTIALIAS)
        #     # combined_img = Image.alpha_composite(original_img.convert("RGBA"), visualized_img_resized.convert("RGBA")).convert("RGB")

        #     # # 保存叠加后的图片
        #     # combined_filename = os.path.join(output_dir, 'combined_image.jpg')
        #     # combined_img.save(combined_filename)
        return x, p_loss, r

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor, r: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x, p_loss, r = self._partition_attn(x, pos_emb, r)
        return x, p_loss, r


# @torch.jit.script
def get_score_index_1d(x: torch.Tensor, b: float) -> torch.Tensor:
    index = torch.nonzero(x >= x.mean() / (1 + b)).view(-1)
    return index

def get_score_index_2d21d(x: torch.Tensor, d: float, b: float, randaug: bool) -> torch.Tensor:
    if x.shape[0] == 1:
        return torch.nonzero(x >= d / (1 + b))[:, 1]
    # if randaug:
    #     b += random.uniform(-0.1, 0.1)
    gt = x >= d / (1 + b)
    index_2d = torch.nonzero(gt)
    index_1d = index_2d[:, 0] * x.shape[-1] + index_2d[:, 1]
    return index_1d

# @torch.jit.script
# def get_score_index_2d21d(x: torch.Tensor, d: float, b: float) -> torch.Tensor:
#     index_1d = torch.arange(x.numel(), device=x.device).view(x.shape)
#     gt = x >= d / (1 + b)
#     index_1d = index_1d[gt]
#     return index_1d


def get_score_index_topk2d(x: torch.Tensor, d: float, b: float, randaug: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # if randaug:
    #     b += random.uniform(-0.2, 0.1)
    gt = x >= d / (1 + b)
    K = torch.sum(gt, dim=1)
    top_indices = torch.topk(x, k=K.max(), dim=1, largest=True, sorted=False)[1]
    arange = torch.arange(0, x.shape[0] * x.shape[1], x.shape[1], device=x.device).view(-1, 1)
    index_2d = torch.nonzero(gt)
    index_1d = index_2d[:, 0] * x.shape[-1] + index_2d[:, 1]
    return (top_indices + arange).view(-1), index_1d, K
    

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


class WindowSA(nn.Module):
    """ Channels-last window multi-head self-attention (B, ..., C) """

    def __init__(
            self,
            dim: int,
            dim_head: int = 32,
            bias: bool = True,
            sub_layers: Optional[List[nn.Module]] = None,
            first_norm: nn.Module = None,):
        super().__init__()
        self.num_heads = dim // dim_head
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.sub_layers = sub_layers
        self.eps = 1e-6
        self.norm =first_norm

    def forward(self, x: torch.Tensor, index_window: torch.Tensor, 
                index_partition: torch.Tensor, blocked_index: torch.Tensor, M: int, K: torch.Tensor) -> torch.Tensor:
        assert self.sub_layers is not None 

        N, C = x.shape[0], x.shape[-1]
        restore_shape = x.shape
        x = x.view(N, -1, C)
        x = self.norm(x)
        if len(index_partition) == 0:
            return x.view(*restore_shape)
        
        X = x.clone()
        x = x[index_window].view(-1, C)
        XX = x.clone()
        x = x[index_partition].view(M, -1, C)

        shortcut = x
        q, k, v = self.qkv(x).view(M, -1, self.num_heads, self.dim_head * 3).transpose(1, 2).chunk(3, dim=3)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn_map = torch.zeros((XX.shape[0], q.shape[2], self.num_heads), device=x.device, dtype=attn.dtype)
        attn_map[index_partition] = attn.transpose(1, 3).reshape(-1, q.shape[2], self.num_heads)
        attn_map[blocked_index] = -1e4
        attn = attn_map[index_partition].view(M, -1, q.shape[2], self.num_heads).transpose(1, 3)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2)
        x = self.proj(x.reshape(M, -1, C))

        for i, layer in enumerate(self.sub_layers):
            if i == 1 or i == 5:
                x = shortcut + layer(x)
                shortcut = x
            else:
                x = layer(x)

        XX[index_partition] = x.view(-1, C)
        XX[blocked_index] = X[index_window].view(-1, C)[blocked_index]
        X[index_window] = XX.view(M, -1, C)
        x = X.view(*restore_shape)
        return x


class WindowSA_WindowPruningOnly(nn.Module):
    """ Channels-last window multi-head self-attention (B, ..., C) """

    def __init__(
            self,
            dim: int,
            dim_head: int = 32,
            bias: bool = True,
            sub_layers: Optional[List[nn.Module]] = None,
            first_norm: nn.Module = None,):
        super().__init__()
        self.num_heads = dim // dim_head
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.sub_layers = sub_layers
        self.eps = 1e-6
        self.norm =first_norm

    def forward(self, x: torch.Tensor, index: torch.Tensor, M: int) -> torch.Tensor:
        assert self.sub_layers is not None 
        N, C = x.shape[0], x.shape[-1]
        restore_shape = x.shape
        x = x.view(N, -1, C)
        x = self.norm(x)
        if len(index) == 0:
            return x.view(*restore_shape)
        X = x.clone()
        x = x[index].view(M, -1, C)
        # x = self.norm(x)
        shortcut = x
        q, k, v = self.qkv(x).view(M, -1, self.num_heads,
                  self.dim_head * 3).transpose(1, 2).chunk(3, dim=3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2)
        x = self.proj(x.reshape(M, -1, C))

        for i, layer in enumerate(self.sub_layers):
            if i == 1 or i == 5:
                x = shortcut + layer(x)
                shortcut = x
            else:
                x = layer(x)
        X[index] = x.view(M, -1, C)
        x = X.view(*restore_shape)
        return x
    

class MyFloorFunc(InplaceFunction):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.floor(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input
    

def floor_div_custom(input, K):
    rem = torch.remainder(input, K)
    out = (input - rem) / K
    return out


def transform_indices(indices_H, indices_W, H, W):
    H_out = len(indices_H)
    W_out = len(indices_W)
    indices_H_tensor = indices_H.view(H_out, 1)
    indices_W_tensor = indices_W.view(1, W_out)
    new_indices = (indices_H_tensor + 1) * (H - 1) * W + (indices_W_tensor + 1) * (W - 1)
    new_indices = new_indices / (H * W - 1) / 2
    return new_indices.view(-1)


def gridsample1d_by2d(input: torch.Tensor, grid: torch.Tensor, padding_mode: str = 'zeros', align_corners: bool = False) -> torch.Tensor:
    shape = grid.shape
    input = input.unsqueeze(-1)  # batch_size * C * L_in * 1
    grid = grid.unsqueeze(-1)  # batch_size * L_out * 1
    grid = torch.stack([-torch.ones_like(grid), grid], dim=-1)
    z = F.grid_sample(input, grid, padding_mode=padding_mode, align_corners=align_corners)
    C = input.shape[1]
    out_shape = [shape[0], C, shape[1]]
    z = z.view(*out_shape)  # batch_size * C * L_out
    return z


class StreamSpreader():
    def __init__(self):
        self.streams = []

    def __call__(self, *tasks):
        if not torch.cuda.is_available():
            return [t() for t in tasks]
        while len(self.streams) < len(tasks):
            self.streams.append(torch.cuda.Stream())
        ret = []
        for s, t in zip(self.streams, tasks):
            with torch.cuda.stream(s):
                ret.append(t())
        return ret


def assert_activation_string(activation_string: Optional[Union[str, Tuple[str, ...], List[str]]]) -> None:
    # Serves as a hacky documentation and sanity check.
    # List of possible activation layer strings that are reasonable:
    # https://github.com/rwightman/pytorch-image-models/blob/a520da9b495422bc773fb5dfe10819acb8bd7c5c/timm/models/layers/create_act.py#L62
    if activation_string is None:
        return
    if isinstance(activation_string, str):
        assert activation_string in ('silu', 'swish', 'mish', 'relu', 'relu6', 'leaky_relu', 'elu', 'prelu', 'celu', 'selu',
                             'gelu', 'sigmoid', 'tanh', 'hard_sigmoid', 'hard_swish', 'hard_mish')
    elif isinstance(activation_string, (tuple, list)):
        for entry in activation_string:
            assert_activation_string(activation_string=entry)
    else:
        raise NotImplementedError


def assert_norm2d_layer_string(norm_layer: Optional[Union[str, Tuple[str, ...], List[str]]]) -> None:
    # Serves as a hacky documentation and sanity check.
    # List of possible norm layer strings that are reasonable:
    # https://github.com/rwightman/pytorch-image-models/blob/4f72bae43be26d9764a08d83b88f8bd4ec3dbe43/timm/models/layers/create_norm.py#L14
    if norm_layer is None:
        return
    if isinstance(norm_layer, str):
        assert norm_layer in ('batchnorm', 'batchnorm2d', 'groupnorm', 'layernorm2d')
    elif isinstance(norm_layer, (tuple, list)):
        for entry in norm_layer:
            assert_norm2d_layer_string(norm_layer=entry)
    else:
        raise NotImplementedError
    

class SelfAttentionCl(nn.Module):
    """ Channels-last window multi-head self-attention (B, ..., C) """

    def __init__(
            self,
            dim: int,
            dim_head: int = 32,
            bias: bool = True,
            sub_layers: Optional[List[nn.Module]] = None,
            partition_size: Tuple[int, int] = None,
            partition_type: PartitionType=None):
        super().__init__()
        self.num_heads = dim // dim_head
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.sub_layers = sub_layers
        self.partition_size = partition_size
        self.partition_type = partition_type
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.sub_layers is not None 
        N, C = x.shape[0], x.shape[-1]
        restore_shape = x.shape

        for i, layer in enumerate(self.sub_layers):
            if i == 2:
                x = shortcut + layer(x)
                shortcut = x
            elif i != 0:
                x = layer(x)  
            else:
                x = layer(x).view(N, -1, C)
                shortcut = x
                q, k, v = self.qkv(x).view(N, -1, self.num_heads,
                self.dim_head * 3).transpose(1, 2).chunk(3, dim=3)
                attn = (q @ k.transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
                x = (attn @ v).transpose(1, 2).reshape(N, -1, C)
                x = self.proj(x)
        x = (shortcut + x)
        return x.view(*restore_shape)
    
@torch.jit.script
def get_unique_index(x: torch.Tensor) -> torch.Tensor:
    _, inverse, counts = torch.unique(x, dim=0, 
        sorted=True, return_inverse=True, return_counts=True)
    inv_sorted = inverse.argsort(stable=True)
    tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
    index = inv_sorted[tot_counts]
    return index


@torch.no_grad()
def get_score_index_2d21d_pad(x: torch.Tensor, HW: int) -> Tuple[torch.Tensor, torch.Tensor]:
    x_means = x.mean(-1).view(-1, 1) / (1 + 1e-1)
    nonzero_indices = torch.nonzero(x >= x_means)   
    unique_indices, counts = torch.unique_consecutive(nonzero_indices[:, 0], return_counts=True)
    max_count = counts.max()
    padded_counts = max_count - counts
    padding_indices = torch.repeat_interleave(unique_indices, padded_counts)
    padding_values = torch.full((padding_indices.shape[0],), HW, dtype=torch.long, device=x.device)
    padding = torch.stack((padding_indices, padding_values), dim=1)
    padded_index = torch.cat((nonzero_indices, padding), dim=0)
    nonzero_indices, unique_indices, padding_indices, padding_values, padding = None, None, None, None, None
    return padded_index, max_count


    # def get_indexes_2d(self, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    #     B, N, h, w = self.B, self.N, self.partition_size[0], self.partition_size[1]
    #     norm2d = torch.norm(scores, dim=[2]).view(B * N, -1)
    #     index2d, hw_max = get_score_index_2d21d_pad(norm2d, h * w)
    #     return index2d, hw_max

    # def get_indexed_indices(self, index1d: torch.Tensor, index2d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    #     B, N, h, w = self.B, self.N, self.partition_size[0], self.partition_size[1]
    #     # indices= torch.linspace(0, B * N * (h * w + 1) - 1, B * N * (h * w + 1), device=index1d.device, dtype=torch.long)
    #     # mask = (index2d[:, 0].unsqueeze(1) == index1d).any(dim=1)
    #     # index = index2d[mask]
    #     # index = index[:, 0] * (h * w + 1) + index[:, 1]
    #     # index1d, index2d = None, None
    #     indices = torch.arange(B * N * (h * w + 1), device=index1d.device)
    #     unique_index1d = torch.unique_consecutive(index1d)
    #     mask = torch.zeros(indices.shape[0], dtype=torch.bool, device=index1d.device)
    #     mask[unique_index1d] = True           
    #     filtered_index2d = index2d[mask[index2d[:, 0]]]
    #     index = filtered_index2d[:, 0] * (h * w + 1) + filtered_index2d[:, 1]    
    #     return indices[index]

    # def get_masked_scores(self, scores: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    #     B, N, h, w = self.B, self.N, self.partition_size[0], self.partition_size[1]
    #     score_mask = self.beta.relu().repeat(B * N * h * w).view_as(scores)
    #     score_mask[index] = self.alpha.relu()
    #     scores = scores.mul_(score_mask)
    #     return scores
        
    # def get_indices_global(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    #     B, H, W, C = x.shape
    #     x = x.view(B, H * W, -1)
    #     indices_BN = torch.linspace(0, B * (H * W + 1) - 1, B * (H * W + 1), device=x.device)
    #     scores = self.to_scores(x).softmax(dim=-1)
    #     norm = torch.norm(scores, dim=[2]).view(B, -1)
    #     index = get_score_index_2d21d_pad(norm, 1.01, H * W)
    #     x = torch.cat([x, torch.zeros([B, 1, C], device=x.device)], dim=1)
    #     scores = torch.cat([scores, torch.zeros([B, 1, C], device=x.device)], dim=1)
    #     indices_BN = indices_BN[index]
    #     gamma = self.gamma.relu()
    #     score_mask = 0.1 * gamma * torch.ones_like(scores).view(-1, self.dim)
    #     score_mask[index] = gamma
    #     x = x * scores.view(x.shape) * score_mask.view(x.shape)
    #     return x, indices_BN
        
    # def distribution_entropy(self, distribution):
    #     probabilities = F.softmax(distribution, dim=-1)
    #     log_probabilities = torch.log(probabilities)
    #     entropy = -torch.sum(probabilities * log_probabilities, dim=-1)
    #     loss = -torch.mean(entropy) 
    #     return loss    