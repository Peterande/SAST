# --------------------------------------------------------
# Copyright (C) 2022 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Official PyTorch implementation of CVPR2022 paper
# A-ViT: Adaptive Tokens for Efficient Vision Transformer
# Hongxu Yin, Arash Vahdat, Jose M. Alvarez, Arun Mallya, Jan Kautz,
# and Pavlo Molchanov
# --------------------------------------------------------

# The following snippets are started from:
# https://github.com/facebookresearch/deit
# &
# https://github.com/rwightman/pytorch-image-models
# Before code is extensively modified to accomodate A-ViT training

import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import PatchEmbed, Mlp, DropPath

from torch.autograd import Variable

import numpy as np
from models.layers.sast.sparseVIT import (
    PartitionAttentionCl,
    nhwC_2_nChw,
    get_downsample_layer_Cf2Cl,
    PartitionType)
from models.layers.rnn import DWSConvLSTM2d

class Masked_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., mask=None, masked_softmax_bias=-1000.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.mask = mask # this is of shape [batch, token_number], where the token number
                         # dimension is indication of token exec.
                         # 0's are the tokens to continue, 1's are the tokens masked out

        self.masked_softmax_bias = masked_softmax_bias

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            # now we need to mask out all the attentions associated with this token
            attn = attn + mask.view(mask.shape[0], 1, 1, mask.shape[1]) * self.masked_softmax_bias
            # this additional bias will make attention associated with this token to be zeroed out
            # this incurs at each head, making sure all embedding sections of other tokens ignore these tokens

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block_ACT(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, args=None, index=-1, num_patches=197):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Masked_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.act_mode = 4
        assert self.act_mode in {1, 2, 3, 4} #now only support 1-extra mlp, or b-position 0 encoding

        self.index=index
        self.args = args

        if self.act_mode == 4:
            # Apply sigmoid on the mean of all tokens to determine whether to continue
            self.sig = torch.sigmoid
        else:
            print('Not supported yet.')
            exit()

    def forward(self, x):

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


    def forward_act(self, x, mask=None):

        debug=False
        analyze_delta = True
        bs, token, dim = x.shape

        if mask is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x*(1-mask).view(bs, token, 1))*(1-mask).view(bs, token, 1), mask=mask))
            x = x + self.drop_path(self.mlp(self.norm2(x*(1-mask).view(bs, token, 1))*(1-mask).view(bs, token, 1)))

        if self.act_mode==4:
            gate_scale, gate_center = 10.0, 30.0
            halting_score_token = self.sig(x[:,:,0] * gate_scale - gate_center)
            # initially first position used for layer halting, second for token
            # now discarding position 1
            halting_score = [-1, halting_score_token]
        else:
            print('Not supported yet.')
            exit()

        return x, halting_score


# Adaptive Vision Transformer
class AVIT(nn.Module):
    """ Vision Transformer with Adaptive Token Capability

    Starting at:
        A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
            - https://arxiv.org/abs/2010.11929

        Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
            - https://arxiv.org/abs/2012.12877

    Extended to:
        Accomodate adaptive token inference
    """

    def __init__(self, dim_in=3, embed_dim=768, depth=12, num_patches=None, spatial_downsample_factor=2,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1, norm_layer=None,
                 act_layer=None, weight_init='', args=None, stage_cfg=None):
        """
        Args:
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """

        super().__init__()
        downsample_cfg = stage_cfg.downsample
        lstm_cfg = stage_cfg.lstm
        attention_cfg = stage_cfg.attention
        self.downsample_cf2cl = get_downsample_layer_Cf2Cl(dim_in=dim_in,
                                                           dim_out=embed_dim,
                                                           downsample_factor=spatial_downsample_factor,
                                                           downsample_cfg=downsample_cfg)
        self.lstm = DWSConvLSTM2d(dim=embed_dim,
                                  dws_conv=lstm_cfg.dws_conv,
                                  dws_conv_only_hidden=lstm_cfg.dws_conv_only_hidden,
                                  dws_conv_kernel_size=lstm_cfg.dws_conv_kernel_size,
                                  cell_update_dropout=lstm_cfg.get('drop_cell_update', 0))
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.Sequential(*[
            Block_ACT(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, args=args, index=i)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()


        print('\nNow this is an ACT DeiT.\n')
        self.eps = 0.01
        print(f'Setting eps as {self.eps}.')

        print('Now re-initializing the halting network bias')
        act_mode = 4
        for block in self.blocks:
            if act_mode == 1:
                # torch.nn.init.constant_(block.act_mlp.fc1.bias.data, -3)
                torch.nn.init.constant_(block.act_mlp.fc2.bias.data, -1. * args.gate_center)

        self.args = args

        print('Now setting up the rho.')
        self.rho = None  # Ponder cost
        self.counter = None  # Keeps track of how many layers are used for each example (for logging)
        self.batch_cnt = 0 # amount of batches seen, mainly for tensorboard

        # for token act part
        self.c_token = None
        self.R_token = None
        self.mask_token = None
        self.rho_token = None
        self.counter_token = None
        self.total_token_cnt = num_patches

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()


    def forward(self, x, h_and_c_previous=None):

        x = self.downsample_cf2cl(x)  # N C H W -> N H W C
        H, W = x.size()[1:3]
        x = x.view(x.size(0), -1, x.size(-1))  # N H W C -> N HW C
        x = self.pos_drop(x + self.pos_embed)

        # now start the act part
        bs = x.size()[0]  # The batch size

        # this part needs to be modified for higher GPU utilization
        if self.c_token is None or bs != self.c_token.size()[0]:
            self.c_token = Variable(torch.zeros(bs, self.total_token_cnt).cuda())
            self.R_token = Variable(torch.ones(bs, self.total_token_cnt).cuda())
            self.mask_token = Variable(torch.ones(bs, self.total_token_cnt).cuda())
            self.rho_token = Variable(torch.zeros(bs, self.total_token_cnt).cuda())
            self.counter_token = Variable(torch.ones(bs, self.total_token_cnt).cuda())

        c_token = self.c_token.clone()
        R_token = self.R_token.clone()
        mask_token = self.mask_token.clone()
        self.rho_token = self.rho_token.detach() * 0.
        self.counter_token = self.counter_token.detach() * 0 + 1.
        # Will contain the output of this residual layer (weighted sum of outputs of the residual blocks)
        output = None
        # Use out to backbone
        out = x
        self.distr_prior_alpha = 0.01
        if self.distr_prior_alpha>0.:
            self.halting_score_layer = []

        for i, l in enumerate(self.blocks):

            # block out all the parts that are not used
            out.data = out.data * mask_token.float().view(bs, self.total_token_cnt, 1)

            # evaluate layer and get halting probability for each sample
            # block_output, h_lst = l.forward_act(out)    # h is a vector of length bs, block_output a 3D tensor
            block_output, h_lst = l.forward_act(out, 1.-mask_token.float())    # h is a vector of length bs, block_output a 3D tensor

            if self.distr_prior_alpha>0.:
                self.halting_score_layer.append(torch.mean(h_lst[1][1:]))

            out = block_output.clone()              # Deep copy needed for the next layer

            _, h_token = h_lst # h is layer_halting score, h_token is token halting score, first position discarded

            # here, 1 is remaining, 0 is blocked
            block_output = block_output * mask_token.float().view(bs, self.total_token_cnt, 1)

            # Is this the last layer in the block?
            if i==len(self.blocks)-1:
                h_token = Variable(torch.ones(bs, self.total_token_cnt).cuda())

            # for token part
            c_token = c_token + h_token
            self.rho_token = self.rho_token + mask_token.float()

            # Case 1: threshold reached in this iteration
            # token part
            reached_token = c_token > 1 - self.eps
            reached_token = reached_token.float() * mask_token.float()
            delta1 = block_output * R_token.view(bs, self.total_token_cnt, 1) * reached_token.view(bs, self.total_token_cnt, 1)
            self.rho_token = self.rho_token + R_token * reached_token

            # Case 2: threshold not reached
            # token part
            not_reached_token = c_token < 1 - self.eps
            not_reached_token = not_reached_token.float()
            R_token = R_token - (not_reached_token.float() * h_token)
            delta2 = block_output * h_token.view(bs, self.total_token_cnt, 1) * not_reached_token.view(bs, self.total_token_cnt, 1)

            self.counter_token = self.counter_token + not_reached_token # These data points will need at least one more layer

            # Update the mask
            mask_token = c_token < 1 - self.eps

            if output is None:
                output = delta1 + delta2
            else:
                output = output + (delta1 + delta2)

        x = self.norm(output)

        x = x.view(bs, H, W, -1)  # N HW C -> N H W C
        x = nhwC_2_nChw(x)  # N H W C -> N C H W
        h_c_tuple = self.lstm(x, h_and_c_previous)
        x = h_c_tuple[0]
        return x, h_c_tuple, reached_token.sum().item()
    