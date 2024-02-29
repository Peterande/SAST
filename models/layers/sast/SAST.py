''' SAST: Scene Adaptive Sparse Transformer for Event-based Object Detection '''
''' Anonymous CVPR 2024 submission supplementary material '''
''' Paper ID: 2476 '''


from enum import Enum, auto
from functools import partial
from typing import Optional, Tuple, List, Type

import math
import torch
from omegaconf import DictConfig
from torch import nn

from .layers import DropPath, LayerNorm
from .layers import get_act_layer, get_norm_layer
from .layers import to_2tuple, _assert


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
        # self.weight = nn.Parameter(torch.ones(1))


    def forward(self, x):
        x = self.net(x)
        return x
        # weight = self.weight.sigmoid()
        # return weight * x + (1 - weight) * x.mean(dim=1, keepdim=True)


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float=1e-5, inplace: bool=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        gamma = self.gamma
        return x.mul_(gamma) if self.inplace else x * gamma
    
    
class SAST_block(nn.Module):
    ''' SAST block contains two SAST layers '''

    def __init__(
            self,
            dim: int,
            attention_cfg: DictConfig,
            first_block: bool=False,
    ):
        super().__init__()
        norm_eps = attention_cfg.get('norm_eps', 1e-5)
        partition_size = attention_cfg.partition_size
        dim_head = attention_cfg.get('dim_head', 32)
        attention_bias = attention_cfg.get('attention_bias', True)
        mlp_act_string = attention_cfg.mlp_activation
        mlp_bias = attention_cfg.get('mlp_bias', True)
        mlp_expand_ratio = attention_cfg.get('mlp_ratio', 4)

        drop_path = attention_cfg.get('drop_path', 0.0)
        drop_mlp = attention_cfg.get('drop_mlp', 0.0)
        ls_init_value = attention_cfg.get('ls_init_value', 1e-5)
        
        if isinstance(partition_size, int):
            partition_size = to_2tuple(partition_size)
        else:
            partition_size = tuple(partition_size)
            assert len(partition_size) == 2
        self.partition_size = partition_size

        norm_layer = partial(get_norm_layer('layernorm'), eps=norm_eps)

        mlp_act_layer = get_act_layer(mlp_act_string)

        sub_layer_params = (ls_init_value, drop_path, mlp_expand_ratio, mlp_act_layer, mlp_bias, drop_mlp)
        
        self_attn_module = MS_WSA
        self.enable_CB = attention_cfg.get('enable_CB', False)

        self.win_attn = self_attn_module(dim,
                                         dim_head=dim_head,
                                         bias=attention_bias,
                                         sub_layer_params=sub_layer_params,
                                         norms=[norm_layer(dim), norm_layer(dim)])

        self.grid_attn = self_attn_module(dim,
                                          dim_head=dim_head,
                                          bias=attention_bias,
                                          sub_layer_params=sub_layer_params,
                                          norms=[norm_layer(dim), norm_layer(dim)])
        if first_block:
            self.to_scores = nn.Linear(dim, dim)
            self.to_controls = PositiveLinear(20, dim, bias=False)
            torch.nn.init.constant_(self.to_controls.weight, 1)
            self.act = nn.ReLU()

        self.amp_value = 2e-4
        self.bounce_value = 1e-3
        self.first_block = first_block
        self.B, self.N, self.dim = None, None, dim

    def window_selection(self, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, h, w = self.B, self.N, self.partition_size[0], self.partition_size[1]
        temp = h * w
        norm_window = (torch.norm(scores, dim=[2, 3], p=1) / temp).softmax(-1) 
        index_window = get_score_index_2d21d(norm_window.view(B, N), 1 / N, self.bounce_value) 
        return index_window
    
    def token_selection(self, scores: torch.Tensor, index_window: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, h, w = self.B, self.N, self.partition_size[0], self.partition_size[1]
        temp = 1
        norm_token = (torch.norm(scores, dim=[3], p=1) / temp).view(B * N, -1)[index_window].softmax(-1)
        index_token, asy_index_partition, K = get_score_index_with_padding(norm_token, 1 / (h * w), self.bounce_value) 
        return index_token, asy_index_partition, K
        
    def _partition_attn(self, x: torch.Tensor, pos_emb: torch.Tensor, r: torch.Tensor, index_list: List) -> Tuple[torch.Tensor, ...]:
        index_count = 0
        self.B = x.shape[0]
        img_size = x.shape[1:3]
        self.N = img_size[0] * img_size[1] // (self.partition_size[0] * self.partition_size[1])

        ''' First SAST Layer '''
        x = x + pos_emb(x)
        x = window_partition(x, self.partition_size).view(self.B, self.N, -1, self.dim) 
        if self.first_block:
            # Scoring Module 
            scale = self.to_controls(r + 1e-6)[:, None, None, :]  
            scores = self.act(self.to_scores(x)) 

            # STP Weighting
            weight = scale.sigmoid() * scores.sigmoid() 
            x = (weight * x).view(self.B * self.N, -1, self.dim) # Weight x use sigmoid scores 

            # Selection Module 
            scale = self.amp_value / scale
            scale[scale==torch.inf] = 0
            scores = scale * scores
            index_window = self.window_selection(scores)
            index_token, asy_index, K = self.token_selection(scores, index_window)
            padding_index = index_token[torch.isin(index_token, asy_index, assume_unique=True, invert=True)] # Get padding index
            index_list1 = [index_window, index_token, padding_index, asy_index, K] # Buffer index list for reusing
        else:
            # Reuse index list
            x = x.view(self.B * self.N, -1, self.dim)
            index_list1, index_list2 = index_list
            index_window, index_token, padding_index, asy_index, K = index_list1
        M = len(index_window)
        
        if len(index_token):
            x = self.win_attn(x, index_window, index_token, padding_index, asy_index, M, self.B, self.enable_CB)
        x = window_reverse(x, self.partition_size, (img_size[0], img_size[1]))
        
        index_count += len(asy_index) // self.B

        ''' Second SAST Layer '''
        if self.first_block:
            # Reuse scores
            scores = window_reverse(scores.view_as(x), self.partition_size, (img_size[0], img_size[1]))
            scores = grid_partition(scores, self.partition_size).view(self.B, self.N, -1, self.dim)

            # Selection Module 
            index_window = self.window_selection(scores) 
            index_token, asy_index, K = self.token_selection(scores, index_window)
            padding_index = index_token[torch.isin(index_token, asy_index, assume_unique=True, invert=True)]
            index_list2 = [index_window, index_token, padding_index, asy_index, K]
        else:
            index_window, index_token, padding_index, asy_index, K = index_list2
        x = x.view(self.B, img_size[0], img_size[1], self.dim)
        x = grid_partition(x, self.partition_size).view(self.B * self.N, -1, self.dim)
        
        M = len(index_window)
        if len(index_token):  
            x = self.grid_attn(x, index_window, index_token, padding_index, asy_index, M, self.B, self.enable_CB)
        x = grid_reverse(x, self.partition_size, (img_size[0], img_size[1]))
        index_count += len(asy_index) // self.B
        return x, index_count, [index_list1, index_list2]

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor, r: torch.Tensor, index_list: List) -> Tuple[torch.Tensor, ...]:
        x, index_count, index_list = self._partition_attn(x, pos_emb, r, index_list)
        return x, index_count, index_list
    

class PositiveLinear(nn.Module):
    ''' Linear layer with positive weights'''
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
    

class MS_WSA(nn.Module):
    ''' Masked Sparse Window (multi-head) Self-Attention (MS-WSA) '''
    ''' Channels-last (B, ..., C) '''

    def __init__(
            self,
            dim: int,
            dim_head: int = 32,
            bias: bool = True,
            sub_layer_params: Optional[List[nn.Module]] = None,
            norms: nn.Module = None,):
        super().__init__()
        self.num_heads = dim // dim_head
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.norm1 =norms[0]

        ls_init_value, drop_path, mlp_expand_ratio, mlp_act_layer, mlp_bias, drop_mlp = sub_layer_params
        self.ls1 = LayerScale(dim=dim, init_values=ls_init_value) if ls_init_value > 0 else nn.Identity()
        self.drop1 = DropPath(drop_prob=drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norms[1]
        self.mlp = MLP(dim=dim, channel_last=True, expansion_ratio=mlp_expand_ratio,
                       act_layer=mlp_act_layer, bias=mlp_bias, drop_prob=drop_mlp)
        self.ls2 = LayerScale(dim=dim, init_values=ls_init_value) if ls_init_value > 0 else nn.Identity()
        self.drop2 = DropPath(drop_prob=drop_path) if drop_path > 0 else nn.Identity()
        self.sub_layers = nn.ModuleList([self.ls1, self.drop1, self.norm2, self.mlp, self.ls2, self.drop2])
            
        self.eps = 1e-6
        

    def forward(self, x: torch.Tensor, index_window: torch.Tensor, 
                index_token: torch.Tensor, padding_index: torch.Tensor, 
                asy_index: torch.Tensor, M: int, B: torch.Tensor, enable_CB: bool) -> torch.Tensor:
        
        N, C = x.shape[0], x.shape[-1]
        restore_shape = x.shape
        x = x.view(N, -1, C)
        x = self.norm1(x)  
        if len(index_token) == 0: # No selected tokens
            return x.view(*restore_shape)
        
        # Gather selected tokens
        X = x.clone() 
        x = x[index_window].view(-1, C) 
        XX = x.clone() 
        x[asy_index] = self.norm2(x[asy_index])  
        shortcut = x[asy_index]  
        x = x[index_token].view(M, -1, C)  

        # Attention
        q, k, v = self.qkv(x).view(M, -1, self.num_heads, self.dim_head * 3).transpose(1, 2).chunk(3, dim=3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Column masking
        attn_map = torch.zeros((XX.shape[0], q.shape[2], self.num_heads), device=x.device, dtype=attn.dtype) 
        attn_map[index_token] = attn.transpose(1, 3).reshape(-1, q.shape[2], self.num_heads) 
        attn_map[padding_index] = -1e4 
        attn = attn_map[index_token].view(M, -1, q.shape[2], self.num_heads).transpose(1, 3) 

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2)
        x = self.proj(x.reshape(M, -1, C))


        XX[index_token] = x.view(-1, C) 
        x = XX[asy_index] 

        for i, layer in enumerate(self.sub_layers):
            if i == 1 or i == 5: # DropPath
                x = shortcut + layer(x)
                shortcut = x
            elif i == 3: # MLP
                x = layer(x)
                if enable_CB: # Context Broadcasting operation
                    temp_X, temp_XX = torch.zeros_like(X), torch.zeros_like(XX)
                    temp_XX[asy_index] = x
                    temp_X[index_window] = temp_XX.view(M, -1, C)
                    temp_X = temp_X.view(B, -1, C)
                    temp_X = (0.5 * temp_X + (1 - 0.5) * temp_X.mean(dim=1, keepdim=True)).view(*restore_shape)
                    x = temp_X[index_window].view(-1, C)[asy_index]
            else: # LayerScale and LayerNorm
                x = layer(x)

        # Scatter selected tokens
        XX[asy_index] = x.view(-1, C)
        XX[padding_index] = X[index_window].view(-1, C)[padding_index]
        X[index_window] = XX.view(M, -1, C) 
        x = X.view(*restore_shape) 
        return x


def get_score_index_2d21d(x: torch.Tensor, d: float, b: float) -> torch.Tensor:
    '''Thresholding for 2D window index selection'''
    if x.shape[0] == 1:
        # Batch size 1 is a special case because torch.nonzero returns a 1D tensor already.
        return torch.nonzero(x >= d / (1 + b))[:, 1]
    gt = x >= d / (1 + b)
    index_2d = torch.nonzero(gt)
    index_1d = index_2d[:, 0] * x.shape[-1] + index_2d[:, 1]
    return index_1d


def get_score_index_with_padding(x: torch.Tensor, d: float, b: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''Thresholding for 2D token index selection (with paddings)'''
    gt = x >= d / (1 + b)
    K = torch.sum(gt, dim=1)
    top_indices = torch.topk(x, k=K.max(), dim=1, largest=True, sorted=False)[1]
    arange = torch.arange(0, x.shape[0] * x.shape[1], x.shape[1], device=x.device).view(-1, 1)
    index_2d = torch.nonzero(gt)
    index_1d = index_2d[:, 0] * x.shape[-1] + index_2d[:, 1]
    return (top_indices + arange).view(-1), index_1d, K
    

def get_non_zero_ratio(x: torch.Tensor) -> torch.Tensor:
    ''' Get the ratio of non-zero elements in each bin for four SAST blocks'''
    '''Input: (B, C, H, W). Output: [(B, C), (B, C), (B, C), (B, C)].'''
    # Downsample to match the receptive field of each SAST block
    x_down_4 = torch.nn.functional.max_pool2d(x.float(), kernel_size=4, stride=4)
    x_down_8 = torch.nn.functional.max_pool2d(x_down_4, kernel_size=2, stride=2)
    x_down_16 = torch.nn.functional.max_pool2d(x_down_8, kernel_size=2, stride=2)
    x_down_32 = torch.nn.functional.max_pool2d(x_down_16, kernel_size=2, stride=2)
    # Count the number of non-zero elements in each bin 
    num_nonzero_1 = torch.sum(torch.sum(x_down_4 != 0, dtype=torch.int16, dim=[2]), dtype=torch.int16, dim=-1)
    num_nonzero_2 = torch.sum(torch.sum(x_down_8 != 0, dtype=torch.int16, dim=[2]), dtype=torch.int16, dim=-1)
    num_nonzero_3 = torch.sum(torch.sum(x_down_16 != 0, dtype=torch.int16, dim=[2]), dtype=torch.int16, dim=-1)
    num_nonzero_4 = torch.sum(torch.sum(x_down_32 != 0, dtype=torch.int16, dim=[2]), dtype=torch.int16, dim=-1)
    result1 = x.shape[0] / x_down_4.numel() * num_nonzero_1.float()
    result2 = x.shape[0] / x_down_8.numel() * num_nonzero_2.float()
    result3 = x.shape[0] / x_down_16.numel() * num_nonzero_3.float()
    result4 = x.shape[0] / x_down_32.numel() * num_nonzero_4.float()
    return [result1, result2, result3, result4]


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


