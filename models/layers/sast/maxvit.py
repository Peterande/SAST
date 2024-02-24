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
import random
from torch.multiprocessing import Pool
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

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
        # self.weight = nn.Parameter(torch.ones(1))


    def forward(self, x):
        x = self.net(x)
        return x
        # weight = self.weight.sigmoid()
        # return weight * x + (1 - weight) * x.mean(dim=1, keepdim=True)


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

        self_attn_module = WindowSA_WindowPruningOnly if self.window_pruning_only else WindowSA      
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
        layer_list1 = [self.ls11, self.drop_path11, 
                      self.norm12, self.mlp1, self.ls12, self.drop_path12]
        
        self.win_attn = self_attn_module(dim,
                                         dim_head=dim_head,
                                         bias=attention_bias,
                                         sub_layers=layer_list1,
                                         first_norm=self.norm1)

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
        layer_list2 = [self.ls21, self.drop_path21, 
                      self.norm22, self.mlp2, self.ls22, self.drop_path22]
        
        self.grid_attn = self_attn_module(dim,
                                          dim_head=dim_head,
                                          bias=attention_bias,
                                          sub_layers=layer_list2,
                                          first_norm=self.norm2)
        if first_block:
            self.to_scores = nn.Linear(dim, dim)
            self.scale = PositiveLinear(20, dim, bias=False)
            torch.nn.init.constant_(self.scale.weight, 1)
            self.act = nn.ReLU()
        
        # self.ls_path = LayerScale(dim=dim, init_values=0.1)
        self.bounce_area = 1e-3
        self.drop = DropPath(drop_prob=0.05)
        
        self.first_block = first_block

        self.B = None  # global batch size
        self.N = None  # global number of windows
        self.dim = dim

    def get_indexes_window(self, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, h, w = self.B, self.N, self.partition_size[0], self.partition_size[1]
        temp = h * w
        norm_window = (torch.norm(scores, dim=[2, 3], p=1) / temp).softmax(-1)
        index_window = get_score_index_2d21d(norm_window.view(B, N), 1 / N, self.bounce_area, randaug=self.training)
        return index_window
    
    def get_indexes_partition(self, scores: torch.Tensor, index_window: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, h, w = self.B, self.N, self.partition_size[0], self.partition_size[1]
        temp = 1
        norm_partition = (torch.norm(scores, dim=[3], p=1) / temp).view(B * N, -1)[index_window].softmax(-1)
        index_partition, asy_index_partition, K = get_score_index_topk2d(norm_partition, 1 / (h * w), self.bounce_area, randaug=self.training)
        return index_partition, asy_index_partition, K
        
    def _partition_attn(self, x: torch.Tensor, pos_emb: nn.Module, r: torch.Tensor, index_list: List) -> Tuple[torch.Tensor, ...]:
        self.B = x.shape[0]
        img_size = x.shape[1:3]
        self.N = img_size[0] * img_size[1] // (self.partition_size[0] * self.partition_size[1])
        # x = x + pos_emb(x)
        pos = window_partition(pos_emb(x), self.partition_size).view(self.B, self.N, -1, self.dim)
        x = window_partition(x, self.partition_size).view(self.B, self.N, -1, self.dim)
        if self.first_block:
            # scale = self.act(self.scale(r + 1e-6))[:, None, None, :]
            scale = self.scale(r + 1e-6)[:, None, None, :]
            scores = self.act(self.to_scores(x))
            # reinit self.to_scores
            # self.to_scores.weight.data.normal_(mean=0.0, std=0.02)
            weight = scale.sigmoid() * scores.sigmoid()
            # x = x + pos.view(self.B, self.N, -1, self.dim)
            x = (weight * x).view(self.B * self.N, -1, self.dim) # weighting x use sigmoid scores.
            # x = x + pos.view(self.B * self.N, -1, self.dim)
            with torch.no_grad():
                scale = 2e-4 / scale
                scale[scale==torch.inf] = 0
                scores = scale * scores
                index_window1 = self.get_indexes_window(scores)
                index1, asy_index1, K1 = self.get_indexes_partition(scores, index_window1)
                blocked_index1 = index1[torch.isin(index1, asy_index1, assume_unique=True, invert=True)]
                index_list1 = [index_window1, index1, blocked_index1, asy_index1, K1]
        else:
            x = x.view(self.B * self.N, -1, self.dim)
            index_list1, index_list2 = index_list
            index_window1, index1, blocked_index1, asy_index1, K1 = index_list1
        M1 = len(index_window1)
        
        
        if len(index1):
            x = self.win_attn(x, index_window1, index1, blocked_index1, asy_index1, M1, self.B)

        img_array2, img_array1 = None, None
        if True and self.dim:
            norm_partition = (torch.norm(scores, dim=[3], p=1)).view(self.N, -1)
            win = torch.zeros([self.B * self.N, self.partition_size[0] * self.partition_size[1], 3], device=x.device)

            # 使用Jet color map
            colormap = plt.get_cmap('jet')
            jet_colors = torch.tensor([colormap(i)[:3] for i in range(256)], dtype=torch.float32).to(norm_partition.device) * 255
            norm_partition -= norm_partition.min()
            norm_partition += 1e-6

            norm_partition_scaled = (255 * norm_partition / norm_partition.max()).long()
            
            win = jet_colors[norm_partition_scaled]
            win = win[:, :, [2, 1, 0]]
            
            img_tensor = window_reverse(win, self.partition_size, (img_size[0], img_size[1]))

            # output_dir = 'vis/token'
            img_array1 = (img_tensor[0]).cpu().numpy().astype(np.uint8)
            # img = Image.fromarray(img_array1)
            # name = 'scores' + str(self.dim) + '.png'
            # filename = os.path.join(output_dir, name)
            # img.save(filename, quality=100)

            win = 255 * torch.ones([self.B * self.N, self.partition_size[0] * self.partition_size[1], 3], device=x.device)
            N = win.shape[0]
            win[index_window1] = torch.tensor([230.0, 230.0, 230.0], device=win.device)
            winsliced = win[index_window1].view(-1, 3)
            temp = winsliced[asy_index1]
            temp = torch.tensor([196.0, 114.0, 70.0], device=x.device)
            # temp = torch.tensor([70.0, 114.0, 196.0], device=x.device) # BGR
            winsliced[asy_index1] = temp
            temp2 = winsliced[blocked_index1]
            # temp2 = torch.tensor([200.0, 200.0, 200.0], device=win.device)
            winsliced[blocked_index1] = temp2
            winsliced = winsliced.view(M1, -1, 3)
            win[index_window1] = winsliced
            win = win.view(N, -1, 3)
            img_tensor = window_reverse(win, self.partition_size, (img_size[0], img_size[1]))

            # output_dir = 'vis/token'
            img_array2 = (img_tensor[0]).cpu().numpy().astype(np.uint8)
            # img = Image.fromarray(img_array2)
            # name = 'tokens' + str(self.dim) + '.png'
            # filename = os.path.join(output_dir, name)
            # img.save(filename, quality=100)

        x = window_reverse(x, self.partition_size, (img_size[0], img_size[1]))

        if self.first_block:       
            scores = window_reverse(scores.view_as(x), self.partition_size, (img_size[0], img_size[1]))
            scores = grid_partition(scores, self.partition_size).view(self.B, self.N, -1, self.dim)
            with torch.no_grad():
                index_window2 = self.get_indexes_window(scores)
                index2, asy_index2, K2 = self.get_indexes_partition(scores, index_window2)
                blocked_index2 = index2[torch.isin(index2, asy_index2, assume_unique=True, invert=True)]
                index_list2 = [index_window2, index2, blocked_index2, asy_index2, K2]
        else:
            index_window2, index2, blocked_index2, asy_index2, K2 = index_list2
        x = x.view(self.B, img_size[0], img_size[1], self.dim)
        x = grid_partition(x, self.partition_size).view(self.B * self.N, -1, self.dim)
        
        M2 = len(index_window2)
        if len(index2):
            x = self.grid_attn(x, index_window2, index2, blocked_index2, asy_index2, M2, self.B)
        x = grid_reverse(x, self.partition_size, (img_size[0], img_size[1]))

        p_loss = (len(index1) + len(index2) - len(blocked_index1) - len(blocked_index2)) // self.B
        return x, p_loss, r, [img_array1, img_array2], [index_list1, index_list2]

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor, r: torch.Tensor, index_list: List) -> Tuple[torch.Tensor, ...]:
        x, p_loss, r, img_array, index_list = self._partition_attn(x, pos_emb, r, index_list)
        return x, p_loss, r, img_array, index_list


class PositiveLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # Apply exponential function to ensure weights are positive
        positive_weights = torch.exp(self.weight)
        return nn.functional.linear(input, positive_weights, self.bias)
    

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
                index_partition: torch.Tensor, blocked_index: torch.Tensor, asy_index: torch.Tensor, M: int, B: torch.Tensor) -> torch.Tensor:
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
        x[asy_index] = self.norm(x[asy_index])
        shortcut = x[asy_index] 
        x = x[index_partition].view(M, -1, C)

        q, k, v = self.qkv(x).view(M, -1, self.num_heads, self.dim_head * 3).transpose(1, 2).chunk(3, dim=3)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn_map = torch.zeros((XX.shape[0], q.shape[2], self.num_heads), device=x.device, dtype=attn.dtype)
        attn_map[index_partition] = attn.transpose(1, 3).reshape(-1, q.shape[2], self.num_heads)
        attn_map[blocked_index] = -1e4
        attn = attn_map[index_partition].view(M, -1, q.shape[2], self.num_heads).transpose(1, 3)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2)
        x = self.proj(x.reshape(M, -1, C))


        XX[index_partition] = x.view(-1, C).float()
        x = XX[asy_index]

        for i, layer in enumerate(self.sub_layers):
            if i == 1 or i == 5:
                x = shortcut + layer(x)
                shortcut = x
            elif i==3:
                x = layer(x)
                # temp_X = torch.zeros_like(X)
                # temp_XX = torch.zeros_like(XX)
                # temp_XX[asy_index] = x
                # temp_X[index_window] = temp_XX.view(M, -1, C)
                # temp_X = temp_X.view(B, -1, C)
                # temp_X = (0.5 * temp_X + (1 - 0.5) * temp_X.mean(dim=1, keepdim=True)).view(*restore_shape)
                # x = temp_X[index_window].view(-1, C)[asy_index]
            else:
                x = layer(x)

        XX[asy_index] = x.view(-1, C)
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