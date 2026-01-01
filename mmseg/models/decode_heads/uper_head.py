# Copyright (c) OpenMMLab. All rights reserved.
from venv import logger

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import SampleList
from ..utils import resize
from .decode_head import BaseDecodeHead
from .psp_head import PPM


@MODELS.register_module()
class UPerHead(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    # in_index=[0, 1, 2, 3]

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,  # 1024 + 4*768
            self.channels, # 768
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def psp_forward(self, inputs):
        # import pdb;pdb.set_trace()
        """Forward function of PSP module."""
        x = inputs[-1]  # [2, 1024, 16, 16]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))  # [2, 1024, 16, 16]--> [2, 768, 16, 16] [2, 768, 16, 16] [2, 768, 16, 16] [2, 768, 16, 16]
        psp_outs = torch.cat(psp_outs, dim=1) # [2, 1024, 16, 16] +  4*[2, 768, 16, 16] --> [2, 4096, 16, 16]
        output = self.bottleneck(psp_outs)    # [2, 4096, 16, 16] -> [2, 768, 16, 16]

        return output

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        # https://blog.csdn.net/yumaomi/article/details/125376320
        inputs = self._transform_inputs(inputs)  # [1, 128, 128, 128] [1, 256, 64, 64] [1, 512, 32, 32] [1, 1024, 16, 16]

        # build laterals,
        # 3xConvModule{Conv2d(X, 768, (1,1), (1,1), bias=False) + bn + ReLU}
        #
        # laterals = [[2, 1024, 128, 128], [2, 1024, 64, 64], [2, 1024, 32, 32]]
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs) # 128-->768 256-->768 512-->768
        ]

        # laterals = [[2, 768, 128, 128], [2, 768, 64, 64], [2, 768, 32, 32] [2, 768, 16, 16]]
        laterals.append(self.psp_forward(inputs)) # psp_forward: [2, 1024, 16, 16] --> [2, 768, 16, 16] # 最后一层的特征

        # build top-down path  # 也就这里去做文章吧？
        # import pdb;pdb.set_trace()
        used_backbone_levels = len(laterals) # 4
        for i in range(used_backbone_levels - 1, 0, -1):  #(3,2,1)
            prev_shape = laterals[i - 1].shape[2:]        # [32,32]
            laterals[i - 1] = laterals[i - 1] + resize(   # laterals[2] =  laterals[2] + resize(laterals[3],32,32)      # 这里把特征拿过来交互啊？
                laterals[i],                              # laterals[1] =  laterals[1] + resize(laterals[2],64,64)
                size=prev_shape,                          # laterals[0] =  laterals[0] + resize(laterals[1],128,128)
                mode='bilinear',
                align_corners=self.align_corners)
        
        # build outputs
        fpn_outs = [                                     # [2, 768, 128, 128], [2, 768, 64, 64], [2, 768, 32, 32]
            self.fpn_convs[i](laterals[i])         
            for i in range(used_backbone_levels - 1)
        ]

        # append psp feature
        fpn_outs.append(laterals[-1])
        # fpn_outs: [[2,768,128,128],[2,768,64,64],[2,768,32,32],[2,1024,16,16]]
        # upsample to the same size
        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        
        # fpn_outs: [[2,768,128,128],[2,768,128,128],[2,768,128,128],[2,768,128,128]]
        fpn_outs = torch.cat(fpn_outs, dim=1) # [2,768*4,128,128]
        # fpn_outs: [2,3072,128,128]

        feats = self.fpn_bottleneck(fpn_outs)  # [2,3072,128,128] --> [2,768,128,128]
        # ConvModule(
        # (conv): Conv2d(4096, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        # (bn): _BatchNormXd(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # (activate): ReLU(inplace=True)
        # )
        # feats.shape: torch.Size([2, 1024, 128, 128])
        return feats

    def forward(self, inputs):
        """Forward function."""

        # import pdb; pdb.set_trace()
        output = self._forward_feature(inputs)  # [2,768,128,128] 
        output = self.cls_seg(output)  # [2,4,128,128]
        return output
    
    # ViT: [2,768,160,160][2,768,80,80][2,768,40,40][2,768,20,20]
    # ConvNeXt: [2,128,160,160][2,256,80,80][2,512,40,40][2,1024,20,20]