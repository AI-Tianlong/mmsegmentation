# --------------------------------------------------------
# PIIP
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from functools import partial
from mmseg.registry import MODELS
from mmengine.logging import print_log
from mmengine.model import BaseModule

try:
    from mmseg.models.ops.deformable_attention.modules import MSDeformAttn

    has_deform_attn = True
except:
    has_deform_attn = False


# mmcv 1.x
# from mmdet.models.builder import BACKBONES
# from mmdet.utils import get_root_logger

@MODELS.register_module()
# class TwoBranch_backbone_mode2(nn.Module):
class TwoBranch_backbone_mode2(BaseModule):
    def __init__(self,
                 branch1_backbone={},
                 branch2_backbone={},
                 ):
        
        super().__init__()

        self.branch1_backbone = MODELS.build(branch1_backbone)  # convnext
        self.branch2_backbone = MODELS.build(branch2_backbone)  # convnext

        # self.branch1_backbone_land_use 不需要梯度
        self.branch1_backbone.eval()
        for param in self.branch1_backbone.parameters():
            param.requires_grad = False


    def forward(self, x):
        self.branch1_backbone.eval()
        branch1_out_list = self.branch1_backbone(x)
        branch2_out_list = self.branch2_backbone(x)

        return branch1_out_list, branch2_out_list