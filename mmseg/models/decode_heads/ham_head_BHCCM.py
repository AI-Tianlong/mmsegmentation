# Copyright (c) OpenMMLab. All rights reserved.
# Originally from https://github.com/visual-attention-network/segnext
# Licensed under the Apache License, Version 2.0 (the "License")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import Tensor


from mmcv.cnn import ConvModule
from mmengine.device import get_device
from typing import List, Tuple
from mmseg.registry import MODELS
from mmseg.utils import SampleList, ConfigType
from mmseg.models.losses import accuracy

from ..utils import resize
from .decode_head import BaseDecodeHead

from typing import List, Tuple


from mmseg.models.losses.atl_hsc_loss import (convert_low_level_label_to_High_level,
                                          L1_L2map, L2_L3map,
                                          MM_5B_18_hiera_structure)


class Matrix_Decomposition_2D_Base(nn.Module):
    """Base class of 2D Matrix Decomposition.

    Args:
        MD_S (int): The number of spatial coefficient in
            Matrix Decomposition, it may be used for calculation
            of the number of latent dimension D in Matrix
            Decomposition. Defaults: 1.
        MD_R (int): The number of latent dimension R in
            Matrix Decomposition. Defaults: 64.
        train_steps (int): The number of iteration steps in
            Multiplicative Update (MU) rule to solve Non-negative
            Matrix Factorization (NMF) in training. Defaults: 6.
        eval_steps (int): The number of iteration steps in
            Multiplicative Update (MU) rule to solve Non-negative
            Matrix Factorization (NMF) in evaluation. Defaults: 7.
        inv_t (int): Inverted multiple number to make coefficient
            smaller in softmax. Defaults: 100.
        rand_init (bool): Whether to initialize randomly.
            Defaults: True.
    """

    def __init__(self,
                 MD_S=1,
                 MD_R=64,
                 train_steps=6,
                 eval_steps=7,
                 inv_t=100,
                 rand_init=True):
        super().__init__()

        self.S = MD_S
        self.R = MD_R

        self.train_steps = train_steps
        self.eval_steps = eval_steps

        self.inv_t = inv_t

        self.rand_init = rand_init

    def _build_bases(self, B, S, D, R, device=None):
        raise NotImplementedError

    def local_step(self, x, bases, coef):
        raise NotImplementedError

    def local_inference(self, x, bases):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    def forward(self, x, return_bases=False):
        """Forward Function."""
        B, C, H, W = x.shape

        # (B, C, H, W) -> (B * S, D, N)
        D = C // self.S
        N = H * W
        x = x.view(B * self.S, D, N)
        if not self.rand_init and not hasattr(self, 'bases'):
            bases = self._build_bases(1, self.S, D, self.R, device=x.device)
            self.register_buffer('bases', bases)

        # (S, D, R) -> (B * S, D, R)
        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R, device=x.device)
        else:
            bases = self.bases.repeat(B, 1, 1)

        bases, coef = self.local_inference(x, bases)

        # (B * S, N, R)
        coef = self.compute_coef(x, bases, coef)

        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = torch.bmm(bases, coef.transpose(1, 2))

        # (B * S, D, N) -> (B, C, H, W)
        x = x.view(B, C, H, W)

        return x


class NMF2D(Matrix_Decomposition_2D_Base):
    """Non-negative Matrix Factorization (NMF) module.

    It is inherited from ``Matrix_Decomposition_2D_Base`` module.
    """

    def __init__(self, args=dict()):
        super().__init__(**args)

        self.inv_t = 1

    def _build_bases(self, B, S, D, R, device=None):
        """Build bases in initialization."""
        if device is None:
            device = get_device()
        bases = torch.rand((B * S, D, R)).to(device)
        bases = F.normalize(bases, dim=1)

        return bases

    def local_step(self, x, bases, coef):
        """Local step in iteration to renew bases and coefficient."""
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ [(B * S, D, R)^T @ (B * S, D, R)] -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # Multiplicative Update
        coef = coef * numerator / (denominator + 1e-6)

        # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
        numerator = torch.bmm(x, coef)
        # (B * S, D, R) @ [(B * S, N, R)^T @ (B * S, N, R)] -> (B * S, D, R)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        # Multiplicative Update
        bases = bases * numerator / (denominator + 1e-6)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        """Compute coefficient."""
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ (B * S, D, R)^T @ (B * S, D, R) -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # multiplication update
        coef = coef * numerator / (denominator + 1e-6)

        return coef


class Hamburger(nn.Module):
    """Hamburger Module. It consists of one slice of "ham" (matrix
    decomposition) and two slices of "bread" (linear transformation).

    Args:
        ham_channels (int): Input and output channels of feature.
        ham_kwargs (dict): Config of matrix decomposition module.
        norm_cfg (dict | None): Config of norm layers.
    """

    def __init__(self,
                 ham_channels=512,
                 ham_kwargs=dict(),
                 norm_cfg=None,
                 **kwargs):
        super().__init__()

        self.ham_in = ConvModule(
            ham_channels, ham_channels, 1, norm_cfg=None, act_cfg=None)

        self.ham = NMF2D(ham_kwargs)

        self.ham_out = ConvModule(
            ham_channels, ham_channels, 1, norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, x):
        enjoy = self.ham_in(x)
        enjoy = F.relu(enjoy, inplace=True)
        enjoy = self.ham(enjoy)
        enjoy = self.ham_out(enjoy)
        ham = F.relu(x + enjoy, inplace=True)

        return ham


class BHCCM_MergeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super().__init__()
        
        self.sigmoid=nn.Sigmoid()
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1) # [2,4,128,128]-->[2,4,1,1]  
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1) # [2,4,128,128]-->[2,4,1,1]  
        self.mlp=nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=out_channels,bias=False), # [2,4,1,1]-->[2,1024,1,1]
            nn.ReLU())
        # L1 spatial_attentation
        self.spatial_conv = nn.Conv2d(in_channels=2, 
                                        out_channels=1, 
                                        kernel_size=7 ,
                                        stride=1,
                                        padding=7//2,
                                        bias=False)
        self.channel_conv_1x1 = nn.Conv2d(in_channels=in_channels, 
                                            out_channels=out_channels, 
                                            kernel_size=1, 
                                            stride=1, 
                                            padding=0, 
                                            bias=False)
        

    def forward(self, convseg_outputs): # 这里传入这个 会不会把他改变了啊。还是稳妥一点，传一个copy进来吧
        # channel
        max_out_channel_att = self.max_pool(convseg_outputs)
        max_out_channel_att = self.mlp(max_out_channel_att.view(max_out_channel_att.size(0),-1))  # [2,4,1,1]-->[2,9]-->[2,18] or 反过来
        avg_out_channel_att = self.avg_pool(convseg_outputs)
        avg_out_channel_att = self.mlp(avg_out_channel_att.view(avg_out_channel_att.size(0),-1)) # [2,4,1,1]-->[2,9]-->[2,18] or 反过来
        channel_att_out = self.sigmoid(max_out_channel_att+avg_out_channel_att) # [2,9]-->[2,18] or 反过来
        # import pdb;pdb.set_trace()
        channel_att_out = channel_att_out.view(channel_att_out.size(0), channel_att_out.size(1),1,1) #[2,9]-->[2,9,1,1]
        # spatial
        max_out_spatial_att, _ = torch.max(convseg_outputs, dim=1, keepdim=True) # [2,4,128,128]-->[2,1,128,128]
        mean_out_spatial_att = torch.mean(convseg_outputs, dim=1, keepdim=True) # [2,4,128,128]-->[2,1,128,128]
        spatial_att_out = torch.cat((max_out_spatial_att, mean_out_spatial_att), dim=1) #[2,2,128,128]
        spatial_att_out = self.sigmoid(self.spatial_conv(spatial_att_out)) #[2,2,128,128]-->[2,1,128,128]
                        # [2,9,1,1] * [2,9,128,128] * [2,1,128,128] #用了广播机制
        # 原始特征  1x1 # 这里有个问题，没用原始特征了啊？
        convseg_outputs_att = self.channel_conv_1x1(convseg_outputs) # [2,4,128,128]-->[2,9,128,128]
        convseg_outputs_att = channel_att_out * convseg_outputs_att * spatial_att_out

        return convseg_outputs_att

@MODELS.register_module()
class LightHamHead_BHCCM(BaseDecodeHead):
    """SegNeXt decode head.

    This decode head is the implementation of `SegNeXt: Rethinking
    Convolutional Attention Design for Semantic
    Segmentation <https://arxiv.org/abs/2209.08575>`_.
    Inspiration from https://github.com/visual-attention-network/segnext.

    Specifically, LightHamHead is inspired by HamNet from
    `Is Attention Better Than Matrix Decomposition?
    <https://arxiv.org/abs/2109.04553>`.

    Args:
        ham_channels (int): input channels for Hamburger.
            Defaults: 512.
        ham_kwargs (int): kwagrs for Ham. Defaults: dict().
    """

    def __init__(self, 
                 ham_channels=512, 
                 ham_kwargs=dict(),
                 ouput_level: str = 'L3',  # 推理时输出的层级，训练时该参数无效
                 num_classes_level_list: List[int] = [4,9,18],    # device
                 results_with_JSPS: bool = True,   # 输出结果，用JSPS严格约束
                 hiera_mode:str = 'xiaorong1',       # 用来修改消融实验的结构的
                 
                 **kwargs):
        

        self.valid_paths = torch.tensor([
            [0,0,0],
            [0,0,1],
            [0,1,2],
            [0,2,3],
            [0,2,4],
            [1,3,5],
            [1,3,6],
            [1,3,7],
            [2,4,8],
            [2,5,9],
            [2,5,10],
            [2,6,11],
            [2,6,12],
            [2,7,13],
            [2,7,14],
            [2,7,15],
            [2,7,16],
            [3,8,17],
        ], dtype=torch.long)


        num_classes = num_classes_level_list[-1]  # 【ATL-LOG】去创建 self.conv_seg
        super().__init__(input_transform='multiple_select', 
                         num_classes=num_classes,
                         **kwargs)


        self.ham_channels = ham_channels

        self.squeeze = ConvModule(
            sum(self.in_channels),
            self.ham_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.hamburger = Hamburger(ham_channels, ham_kwargs, **kwargs)

        self.align = ConvModule(
            self.ham_channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        
        #============= 创建Hiera需要用到的模块。=======================
        self.test_output_level = ouput_level  # 测试推理时输出的层级
        self.results_with_JSPS = results_with_JSPS
        self.hiera_mode = hiera_mode
        if isinstance(num_classes_level_list, list):
            self.num_classes_level_list = num_classes_level_list  # [5,9,10]


            if self.hiera_mode == 'xiaorong2':
                # 消融实验2  # 多层级间没有交互。只是输出三个通道，然后去独立计算CEloss。
                self.conv_seg_L1 = nn.Conv2d(self.channels, num_classes_level_list[0], kernel_size=1) #(1024-->4)
                self.conv_seg_L2 = nn.Conv2d(self.channels, num_classes_level_list[1], kernel_size=1) #(1024-->9)
                self.conv_seg_L3 = self.conv_seg                                                      #(1024-->9)
            
            elif self.hiera_mode == 'xiaorong3':
                # 消融实验3  # 多层级间只有从粗到细交互，然后独立计算CELoss。
                self.conv_seg_L1 = nn.Conv2d(self.channels, num_classes_level_list[0], kernel_size=1) #(1024-->5)
                self.conv_seg_L2 = nn.Conv2d(self.channels, num_classes_level_list[1], kernel_size=1)
                self.conv_seg_L3 = self.conv_seg
                # stage1: coarse to fine 4-->9 | 4->18 9->18
                self.stage1_1to2_MB = BHCCM_MergeBlock(num_classes_level_list[0], num_classes_level_list[1])# For L2
                self.stage1_w12 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
                self.stage1_w22 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
                
                self.stage1_1to3_MB = BHCCM_MergeBlock(num_classes_level_list[0], num_classes_level_list[2])# For L3
                self.stage1_2to3_MB = BHCCM_MergeBlock(num_classes_level_list[1], num_classes_level_list[2])
                self.stage1_w13 = nn.Parameter(torch.tensor(1.0), requires_grad=True) 
                self.stage1_w23 = nn.Parameter(torch.tensor(1.0), requires_grad=True) 
                self.stage1_w33 = nn.Parameter(torch.tensor(1.0), requires_grad=True) 
                

            elif self.hiera_mode == 'xiaorong4':
               # 消融实验4  # 多层级间只有从细到粗交互，然后独立计算CELoss。
                self.conv_seg_L1 = nn.Conv2d(self.channels, num_classes_level_list[0], kernel_size=1) #(1024-->5)
                self.conv_seg_L2 = nn.Conv2d(self.channels, num_classes_level_list[1], kernel_size=1)
                self.conv_seg_L3 = self.conv_seg

                # stage2: fine to coarse 18-->9 | 18-->4  9-->4
                self.stage2_3to2_MB = BHCCM_MergeBlock(num_classes_level_list[2], num_classes_level_list[1])# For L2
                self.stage2_y32 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
                self.stage2_y22 = nn.Parameter(torch.tensor(1.0), requires_grad=True)

                self.stage2_3to1_MB = BHCCM_MergeBlock(num_classes_level_list[2], num_classes_level_list[0])# For L1
                self.stage2_2to1_MB = BHCCM_MergeBlock(num_classes_level_list[1], num_classes_level_list[0])# For L1
                self.stage2_y31 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
                self.stage2_y21 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
                self.stage2_y11 = nn.Parameter(torch.tensor(1.0), requires_grad=True)

            elif self.hiera_mode == 'xiaorong5': # BHCCM
                # 消融实验5  # 多层级间双向交互，然后用L_HSC作为损失函数。
                self.conv_seg_L1 = nn.Conv2d(self.channels, num_classes_level_list[0], kernel_size=1) #[2,1024,128,128]->[2,4,128,128]
                self.conv_seg_L2 = nn.Conv2d(self.channels, num_classes_level_list[1], kernel_size=1) #(1024-->9)
                self.conv_seg_L3 = self.conv_seg                                                      #(1024-->18) # 默认的conv_seg
                
                # stage1: coarse to fine 4-->9 | 4->18 9->18
                self.stage1_1to2_MB = BHCCM_MergeBlock(num_classes_level_list[0], num_classes_level_list[1])# For L2
                self.stage1_w12 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
                self.stage1_w22 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
                
                self.stage1_1to3_MB = BHCCM_MergeBlock(num_classes_level_list[0], num_classes_level_list[2])# For L3
                self.stage1_2to3_MB = BHCCM_MergeBlock(num_classes_level_list[1], num_classes_level_list[2])
                self.stage1_w13 = nn.Parameter(torch.tensor(1.0), requires_grad=True) 
                self.stage1_w23 = nn.Parameter(torch.tensor(1.0), requires_grad=True) 
                self.stage1_w33 = nn.Parameter(torch.tensor(1.0), requires_grad=True) 
                
                # stage2: fine to coarse 18-->9 | 18-->4  9-->4
                self.stage2_3to2_MB = BHCCM_MergeBlock(num_classes_level_list[2], num_classes_level_list[1])# For L2
                self.stage2_y32 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
                self.stage2_y22 = nn.Parameter(torch.tensor(1.0), requires_grad=True)

                self.stage2_3to1_MB = BHCCM_MergeBlock(num_classes_level_list[2], num_classes_level_list[0])# For L1
                self.stage2_2to1_MB = BHCCM_MergeBlock(num_classes_level_list[1], num_classes_level_list[0])# For L1
                self.stage2_y31 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
                self.stage2_y21 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
                self.stage2_y11 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
                 
            elif self.hiera_mode == 'xiaorong6': # BHCCM
                # 消融实验6  # 多层级间双向交互，然后最后一个的输出，是原始输出特征，叠加交互后的特征，然后用L_HSC作为损失函数。
                self.conv_seg_L1 = nn.Conv2d(self.channels, num_classes_level_list[0], kernel_size=1) #[2,1024,128,128]->[2,4,128,128]
                self.conv_seg_L2 = nn.Conv2d(self.channels, num_classes_level_list[1], kernel_size=1) #(1024-->9)
                self.conv_seg_L3 = self.conv_seg                                                      #(1024-->18) # 默认的conv_seg
                
                # stage1: coarse to fine 4-->9 | 4->18 9->18
                self.stage1_1to2_MB = BHCCM_MergeBlock(num_classes_level_list[0], num_classes_level_list[1])# For L2
                self.stage1_w12 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
                self.stage1_w22 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
                
                self.stage1_1to3_MB = BHCCM_MergeBlock(num_classes_level_list[0], num_classes_level_list[2])# For L3
                self.stage1_2to3_MB = BHCCM_MergeBlock(num_classes_level_list[1], num_classes_level_list[2])
                self.stage1_w13 = nn.Parameter(torch.tensor(1.0), requires_grad=True) 
                self.stage1_w23 = nn.Parameter(torch.tensor(1.0), requires_grad=True) 
                self.stage1_w33 = nn.Parameter(torch.tensor(1.0), requires_grad=True) 
                
                # stage2: fine to coarse 18-->9 | 18-->4  9-->4
                self.stage2_3to2_MB = BHCCM_MergeBlock(num_classes_level_list[2], num_classes_level_list[1])# For L2
                self.stage2_y32 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
                self.stage2_y22 = nn.Parameter(torch.tensor(1.0), requires_grad=True)

                self.stage2_3to1_MB = BHCCM_MergeBlock(num_classes_level_list[2], num_classes_level_list[0])# For L1
                self.stage2_2to1_MB = BHCCM_MergeBlock(num_classes_level_list[1], num_classes_level_list[0])# For L1
                self.stage2_y31 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
                self.stage2_y21 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
                self.stage2_y11 = nn.Parameter(torch.tensor(1.0), requires_grad=True) 

                self.stage2_z11 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
                self.stage2_z22 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
                self.stage2_z33 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
               
            else:
                raise ValueError(f'不支持的 hiera_mode: {self.hiera_mode}, 请检查消融实验配置')


    # ====================== Hiera Module ======================
    def cls_seg_hiear(self, decode_head_outputs: Tensor, conv_seg: nn.Module):
        if self.dropout is not None:
            feat = self.dropout(decode_head_outputs)
        output = conv_seg(decode_head_outputs)
        return output

    def hiera_module(self, inputs: List[Tensor], decode_head_outputs: Tensor):
        if self.hiera_mode == 'xiaorong2':
            # 消融实验2  # 多层级间没有交互。只是输出三个通道，然后去独立计算CEloss。
            # F^i_{in}
            L1_in = self.cls_seg_hiear(decode_head_outputs, self.conv_seg_L1) # [2,1024,128,128]->[2,4,128,128]   F^1_{in}=F^1_{out}
            L2_in = self.cls_seg_hiear(decode_head_outputs, self.conv_seg_L2) # [2,1024,128,128]->[2,9,128,128]   F^2_{in}=F^2_{out}
            L3_in = self.cls_seg_hiear(decode_head_outputs, self.conv_seg_L3) # [2,1024,128,128]->[2,18,128,128]  F^3_{in}=F^3_{out}
            # F^i_{mid}
            L1_mid = L1_in
            L2_mid = L2_in
            L3_mid = L3_in
            # F^i_{out}
            L1_out = L1_mid
            L2_out = L2_mid
            L3_out = L3_mid
            output_list = [L1_out, L2_out, L3_out]
            return output_list
        
        elif self.hiera_mode == 'xiaorong3':
            # 消融实验3  # 多层级间只有从粗到细交互，然后独立计算CELoss。
            L1_in = self.cls_seg_hiear(decode_head_outputs, self.conv_seg_L1) # [2,1024,128,128]->[2,4,128,128]  F^1_{in}
            L2_in = self.cls_seg_hiear(decode_head_outputs, self.conv_seg_L2) # [2,1024,128,128]->[2,9,128,128]  F^2_{in}
            L3_in = self.cls_seg_hiear(decode_head_outputs, self.conv_seg_L3) # [2,1024,128,128]->[2,18,128,128] F^3_{in}
            
            L1_mid  = L1_in
            L2_mid  = self.stage1_w12 * self.stage1_1to2_MB(L1_in) + self.stage1_w22 * L2_in 
            L3_mid  = self.stage1_w13 * self.stage1_1to3_MB(L1_in) + \
                      self.stage1_w23 * self.stage1_2to3_MB(L2_in) + \
                      self.stage1_w33 * L3_in
            
            L1_out = L1_mid
            L2_out = L2_mid
            L3_out = L3_mid
            
            output_list = [L1_out, L2_out, L3_out]
            return output_list
        
        elif self.hiera_mode == 'xiaorong4':
            # 消融实验4  # 多层级间只有从细到粗交互，然后独立计算CELoss。
            L1_in = self.cls_seg_hiear(decode_head_outputs, self.conv_seg_L1) # [2,1024,128,128]->[2,4,128,128]  F^1_{in}
            L2_in = self.cls_seg_hiear(decode_head_outputs, self.conv_seg_L2) # [2,1024,128,128]->[2,9,128,128]  F^2_{in}
            L3_in = self.cls_seg_hiear(decode_head_outputs, self.conv_seg_L3) # [2,1024,128,128]->[2,18,128,128] F^3_{in}
            
            L1_mid  = L1_in
            L2_mid  = L2_in
            L3_mid  = L3_in
                         
            L1_out = self.stage2_y31*self.stage2_3to1_MB(L3_mid) + \
                     self.stage2_y21*self.stage2_2to1_MB(L2_mid) + \
                     self.stage2_y11*L1_mid
            L2_out = self.stage2_y32*self.stage2_3to2_MB(L3_mid) + \
                     self.stage2_y22*L2_mid
            L3_out = L3_mid
            
            output_list = [L1_out, L2_out, L3_out]
            return output_list

        elif self.hiera_mode == 'xiaorong5':
            # 消融实验5  # 多层级间双向交互，然后用L_HSC loss。
            # original feature
            L1_in = self.cls_seg_hiear(decode_head_outputs, self.conv_seg_L1) # [2,1024,128,128]->[2,4,128,128]
            L2_in = self.cls_seg_hiear(decode_head_outputs, self.conv_seg_L2) # [2,1024,128,128]->[2,9,128,128]
            L3_in = self.cls_seg_hiear(decode_head_outputs, self.conv_seg_L3) # [2,1024,128,128]->[2,18,128,128]

            # stage1 coarse to fine
            L1_mid  = L1_in
            L2_mid  = self.stage1_w12 * self.stage1_1to2_MB(L1_in) + self.stage1_w22 * L2_in 
            L3_mid  = self.stage1_w13 * self.stage1_1to3_MB(L1_in) + \
                      self.stage1_w23 * self.stage1_2to3_MB(L2_in) + \
                      self.stage1_w33 * L3_in

            # stage2: fine to coarse
            L1_out = self.stage2_y31*self.stage2_3to1_MB(L3_mid) + \
                     self.stage2_y21*self.stage2_2to1_MB(L2_mid) + \
                     self.stage2_y11*L1_mid
            L2_out = self.stage2_y32*self.stage2_3to2_MB(L3_mid) + \
                     self.stage2_y22*L2_mid
            L3_out = L3_mid
            
            output_list = [L1_out, L2_out, L3_out]  # TODO:双向信息融合完的特征。用不用把这个融合完的特征，再和没融合之前的特征，做个交互/叠加？

            # [[2, 4, 160, 160],[2, 9, 160, 160],[2,18,160,160]]
            return output_list

        elif self.hiera_mode == 'xiaorong6':
            # 消融实验6  # 多层级间双向交互，把原始特征，再给一个给交互后的，然后一起输出用L_HSC loss。
            # original feature
            L1_in = self.cls_seg_hiear(decode_head_outputs, self.conv_seg_L1) # [2,1024,128,128]->[2,4,128,128]
            L2_in = self.cls_seg_hiear(decode_head_outputs, self.conv_seg_L2) # [2,1024,128,128]->[2,9,128,128]
            L3_in = self.cls_seg_hiear(decode_head_outputs, self.conv_seg_L3) # [2,1024,128,128]->[2,18,128,128]

            # stage1 coarse to fine
            L1_mid  = L1_in
            L2_mid  = self.stage1_w12 * self.stage1_1to2_MB(L1_in) + self.stage1_w22 * L2_in 
            L3_mid  = self.stage1_w13 * self.stage1_1to3_MB(L1_in) + \
                      self.stage1_w23 * self.stage1_2to3_MB(L2_in) + \
                      self.stage1_w33 * L3_in

            # stage2: fine to coarse
            L1_out = self.stage2_y31*self.stage2_3to1_MB(L3_mid) + \
                     self.stage2_y21*self.stage2_2to1_MB(L2_mid) + \
                     self.stage2_y11*L1_mid
            L2_out = self.stage2_y32*self.stage2_3to2_MB(L3_mid) + \
                     self.stage2_y22*L2_mid
            L3_out = L3_mid
            
            L1_out = L1_in + self.stage2_z11 * L1_out
            L2_out = L2_in + self.stage2_z22 * L2_out
            L3_out = L3_in + self.stage2_z33 * L3_out


            output_list = [L1_out, L2_out, L3_out]  # TODO:双向信息融合完的特征。用不用把这个融合完的特征，再和没融合之前的特征，做个交互/叠加？

            # [[2, 4, 160, 160],[2, 9, 160, 160],[2,18,160,160]]
            return output_list       
        else:
            raise ValueError(f'不支持的 hiera_mode: {self.hiera_mode}, 请检查消融实验配置')
        
    def forward(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)

        inputs = [
            resize(
                level,
                size=inputs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners) for level in inputs
        ]

        inputs = torch.cat(inputs, dim=1)
        # apply a conv block to squeeze feature map
        x = self.squeeze(inputs)
        # apply hamburger module
        x = self.hamburger(x)

        # apply a conv block to align feature map
        output = self.align(x)      # [2,512,80,80]

       
        
        output_list = self.hiera_module(inputs, output)  # [2,4,160,160] [2,9,160,160] [2,18,160,160]
        return output_list
    
        # output = self.cls_seg(output) # [2,18,80,80]
        # return output

    
    # =================== Hiera 修改 LOSS 和 predict 方式 ===============

    #  ==================================================
    def loss_by_feat(self, 
                     seg_logits: Tensor, # forward 输出的结果
                     batch_data_samples: SampleList) -> dict:
        """Compute segmentation loss.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if isinstance(seg_logits, list) and len(seg_logits) == 3:
            seg_logits = seg_logits
        else:
            raise TypeError(f'请检查decode upernet forward的输出！')

        seg_label = self._stack_batch_gt(batch_data_samples)  # [2,1,512,512]
        loss = dict()
        # 原来是直接把，[2,65,128,128]---双线性插值--->[2,65,512,512]

        if isinstance(seg_logits, list) and len(seg_logits) == 3:
            for i in range(len(seg_logits)):
                seg_logits[i] = resize(
                    input=seg_logits[i],
                    size=seg_label.shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners)
                # print(seg_logits[i].shape)
        elif isinstance(seg_logits, torch.Tensor) and seg_logits.shape[1]==sum(self.num_classes_level_list):
            seg_logits = resize(
                    input=seg_logits,
                    size=seg_label.shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners)
        else:
            raise TypeError(
                f"`seg_logits` 类型不正确，期望为长度为 3 的列表或满足指定通道数的 Tensor，"
                f"但收到类型：{type(seg_logits)}")

        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None # 【ATL-LOG】走这里

        seg_label = seg_label.squeeze(1)  # [2,1,512,512]-->[2,512,512]

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        
        # 如果loss_decode的config是多个dict的话，则构造的时候，
        # self.loss_decode.append(MODELS.build(loss))
        # 否则只是 self.loss_decode = MODELS.build(loss_decode)
        
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:  # loss['atl_loss_ce'],log就打印decode.atl_loss_ce
                # import pdb; pdb.set_trace()
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logits, # output_list[L1,L2,L3]
                    seg_label,  # L3级别的label
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
                # print(loss)
                # import pdb; pdb.set_trace()
            else:
                # pdb.set_trace()
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logits, 
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)

        # 算精度的时候
        # 融合 L1+L2+L3 三层
        if isinstance(seg_logits, list) and len(seg_logits) == 3:
            # 合并L1 L2 L3 级的推理结果
            seg_logits_L1 = seg_logits[0] # L1的特征图
            seg_logits_L2 = seg_logits[1] # L2的特征图
            seg_logits_L3 = seg_logits[2] # L3的特征图
        else:
            raise TypeError(f'seg_logits 应该是个 list ',f'但是得到了个 {type(seg_logits)}')

        # L1、L2、L3级别的精度
        seg_label_list = convert_low_level_label_to_High_level(seg_label, MM_5B_18_hiera_structure)
        loss['acc_seg_L1'] = accuracy(seg_logits_L1, seg_label_list[0], ignore_index=self.ignore_index)
        loss['acc_seg_L2'] = accuracy(seg_logits_L2, seg_label_list[1], ignore_index=self.ignore_index)
        loss['acc_seg_L3'] = accuracy(seg_logits_L3, seg_label_list[2], ignore_index=self.ignore_index)
        loss['acc_seg'] = accuracy(seg_logits_L3, seg_label, ignore_index=self.ignore_index)


        # if self.results_path_merge:
        #     path_merge_mask = self.results_path_merge_func(seg_logits) # 选择最优路径的mask，获得的是mask，而不是seglogits？
        #     # seg_logits = seg_logits[2] # 输出融合后L3的特征图, 去计算精度
        #     path_merge_mask_L1 = path_merge_mask[:,0:1,:,:]
        #     path_merge_mask_L2 = path_merge_mask[:,1:2,:,:]
        #     path_merge_mask_L3 = path_merge_mask[:,2:3,:,:]

        # # import pdb; pdb.set_trace()
        if self.results_with_JSPS:
            JSPS_preds_L1, JSPS_preds_L2, JSPS_preds_L3, best_path_idx, _ = self.jsps_inference_from_logits(pred_seg_logits=seg_logits, valid_paths=self.valid_paths)
            
            JSPS_preds_L1 = JSPS_preds_L1.unsqueeze(1)
            JSPS_preds_L2 = JSPS_preds_L2.unsqueeze(1)
            JSPS_preds_L3 = JSPS_preds_L3.unsqueeze(1)

            loss['acc_seg_JSPS_preds_L1'] = accuracy_path_merge_results(JSPS_preds_L1, seg_label_list[0], ignore_index=self.ignore_index)
            loss['acc_seg_JSPS_preds_L2'] = accuracy_path_merge_results(JSPS_preds_L2, seg_label_list[1], ignore_index=self.ignore_index)
            loss['acc_seg_JSPS_preds_L3'] = accuracy_path_merge_results(JSPS_preds_L3, seg_label_list[2], ignore_index=self.ignore_index)

        # import pdb; pdb.set_trace()\

        # import pdb; pdb.set_trace()
        return loss
    
    def predict_by_feat(self, seg_logits: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Transform a batch of output seg_logits to the input shape.  # 缩放！

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        if self.test_output_level is None:
            self.test_output_level = 'L3'

        # 这里有问题呀，必须输出的是一个层级的结构，不然没办法后处理

        if isinstance(batch_img_metas[0]['img_shape'], torch.Size):
            # slide inference
            size = batch_img_metas[0]['img_shape']
        elif 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape'][:2]
        else:
            size = batch_img_metas[0]['img_shape']

        if isinstance(seg_logits, list) and len(seg_logits) == 3:
            for i in range(len(seg_logits)):
                seg_logits[i] = resize(
                    input=seg_logits[i],
                    size=size,
                    mode='bilinear',
                    align_corners=self.align_corners)
                # print(seg_logits[i].shape)
        elif isinstance(seg_logits, torch.Tensor) and seg_logits.shape[1]==sum(self.num_classes_level_list):
            seg_logits = resize(
                    input=seg_logits,
                    size=size.shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners)

        if self.results_with_JSPS:
            JSPS_preds_L1, JSPS_preds_L2, JSPS_preds_L3, best_path_idx, _ = self.jsps_inference_from_logits(pred_seg_logits=seg_logits, valid_paths=self.valid_paths)

            B, H, W = JSPS_preds_L1.shape
            final_pred = torch.zeros(B, len(seg_logits), H, W, dtype=torch.long, device=JSPS_preds_L1.device).fill_(255)
            final_pred[:, 0] = JSPS_preds_L1
            final_pred[:, 1] = JSPS_preds_L2
            final_pred[:, 2] = JSPS_preds_L3

            return (seg_logits, final_pred) # [1,18,640,640], pred_mask # 传递给了Encder和Decoder的 predict
        # 这里是不是应该写在后处理里啊？  写在这里好像不太对，应为post要的是seglogits然后处理。
        return seg_logits  # 输入 seg_logits的list，然后去自动处理成三个mask
    
    def jsps_inference_from_logits(self, pred_seg_logits, valid_paths: torch.Tensor):
        device = pred_seg_logits[0].device
        valid_paths = valid_paths.to(device=device, dtype=torch.long)

        logits1, logits2, logits3 = pred_seg_logits
        B, _, H, W = logits1.shape
        T = valid_paths.shape[0]

        # 1) log-prob per level: (B,C,H,W) -> (B,H,W,C)
        logp1 = F.log_softmax(logits1, dim=1).permute(0, 2, 3, 1)
        logp2 = F.log_softmax(logits2, dim=1).permute(0, 2, 3, 1)
        logp3 = F.log_softmax(logits3, dim=1).permute(0, 2, 3, 1)

        # 2) enumerate paths and sum log probs: path_scores (B,H,W,T)
        path_scores = torch.zeros((B, H, W, T), device=device, dtype=logp1.dtype)

        idx1 = valid_paths[:, 0].view(1, 1, 1, T).expand(B, H, W, T)
        idx2 = valid_paths[:, 1].view(1, 1, 1, T).expand(B, H, W, T)
        idx3 = valid_paths[:, 2].view(1, 1, 1, T).expand(B, H, W, T)

        path_scores += torch.gather(logp1, dim=-1, index=idx1)
        path_scores += torch.gather(logp2, dim=-1, index=idx2)
        path_scores += torch.gather(logp3, dim=-1, index=idx3)

        # 3) best path per pixel
        best_path_idx = path_scores.argmax(dim=-1)  # (B,H,W)

        # 4) decode per-level predictions from best path
        preds_L1 = valid_paths[best_path_idx, 0]  # (B,H,W)
        preds_L2 = valid_paths[best_path_idx, 1]
        preds_L3 = valid_paths[best_path_idx, 2]

        return preds_L1, preds_L2, preds_L3, best_path_idx, path_scores

def accuracy_path_merge_results(pred, target, topk=1, thresh=None, ignore_index=None):
    """Calculate accuracy according to the prediction and target.

    Args:
        pred (torch.Tensor): The model prediction, shape (N, num_class, ...)
        target (torch.Tensor): The target of each prediction, shape (N, , ...)
        ignore_index (int | None): The label index to be ignored. Default: None
        topk (int | tuple[int], optional): If the predictions in ``topk``
            matches the target, the predictions will be regarded as
            correct ones. Defaults to 1.
        thresh (float, optional): If not None, predictions with scores under
            this threshold are considered incorrect. Default to None.

    Returns:
        float | tuple[float]: If the input ``topk`` is a single integer,
            the function will return a single float as accuracy. If
            ``topk`` is a tuple containing multiple integers, the
            function will return a tuple containing accuracies of
            each ``topk`` number.
    """
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    pred = pred.transpose(0, 1) #[2,1,640,640]-->[1,2,640,640]
    correct = pred.eq(target.unsqueeze(0).expand_as(pred))
    if ignore_index is not None:
        correct = correct[:, target != ignore_index]
    res = []
    eps = torch.finfo(torch.float32).eps
    for k in topk:
        # Avoid causing ZeroDivisionError when all pixels
        # of an image are ignored
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True) + eps
        if ignore_index is not None:
            total_num = target[target != ignore_index].numel() + eps
        else:
            total_num = target.numel() + eps
        res.append(correct_k.mul_(100.0 / total_num))
    return res[0] if return_single else res

