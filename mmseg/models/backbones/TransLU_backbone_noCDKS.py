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
from torch import Tensor
from typing import List, Tuple
from mmseg.utils import SampleList, ConfigType

from mmseg.models.backbones.convnext import ConvNeXt
from mmseg.models.backbones.deit3 import DeiT3
from .piip_modules import deform_inputs_1_vit, deform_inputs_2_vit, TwoBranchInteractionBlock

try:
    from mmseg.models.ops.deformable_attention.modules import MSDeformAttn

    has_deform_attn = True
except:
    has_deform_attn = False


@MODELS.register_module()
# class TwoBranch_backbone_mode2(nn.Module):
class TransLU_backbone_noCDKS(BaseModule):
    def __init__(self,
                branch1_backbone={},
                branch2_backbone={},
                ):
        
        super().__init__()

        branch1_backbone_cfg = branch1_backbone.copy()
        branch2_backbone_cfg = branch2_backbone.copy()
    
        # 构建两个分支的backbone模型，并用初始化, 这里貌似并不会去主动初始化了，应该去显示的初始化
        self.branch1_backbone = MODELS.build(branch1_backbone_cfg)  # convnext
        self.branch2_backbone = MODELS.build(branch2_backbone_cfg)  # convnext

        # 冻结 branch2 的参数，不需要更新

        # 在此确保，branch1_backbone 的参数不会被更新，仅作推理和交互，并提供特征图 在_init_就已经有了，这里不需要了吧？
        self.branch2_backbone.eval()
        for param in self.branch2_backbone.parameters():
            param.requires_grad = False


    def forward(self, x):
        
        branch1_outpit_list = self.branch1_backbone(x)
        branch2_output_list = self.branch2_backbone(x) # [4,10,512,512]-->[1,128,128,128] [1,256,64,64] [1,512,32,32] [1,1024,16,16]

        return branch1_outpit_list, branch2_output_list
        

