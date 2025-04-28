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

from mmengine.model import BaseModule

@MODELS.register_module()
# class TwoBranchd_decode_head_mode2(nn.Module):
class TwoBranch_decode_head_mode2(BaseModule):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    # in_index=[0, 1, 2, 3]

    def __init__(self, 
                 branch1_decode_head={},
                 branch2_decode_head={},
                 ):
        super().__init__()
        

        self.branch1_decode_head = MODELS.build(branch1_decode_head)  # convnext
        self.branch2_decode_head = MODELS.build(branch2_decode_head)  # convnext

        # self.branch1_decode_head_land_use 不需要梯度
        self.branch1_decode_head.eval()
        for param in self.branch1_decode_head.parameters():
            param.requires_grad = False


        # 这里除了传递inputs参数，还应该把batch_img_metas和标签传进来。
    def forward(self, inputs): # 这个inputs 是两个list
        
        branch1_inputs = inputs[0]
        branch2_inputs = inputs[1]

        # forward / predict:forward-->predict_by_feat
        self.branch1_decode_head.eval()
        branch1_land_use_output = self.branch1_decode_head.forward(branch1_inputs) # 没有predict的参数啊  

        branch2_new_task_output = self.branch2_decode_head.forward(branch2_inputs)
        
        # 在这里，应该把branch1 和 branch2 的结构拿出来，像piip一样。

        return branch1_land_use_output, branch2_new_task_output
