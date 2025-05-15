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
class UPerHeadWithGate_iSAID(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    # in_index=[0, 1, 2, 3]

    def __init__(self, 
                 pool_scales=(1, 2, 3, 6), 
                 dataset = 'crop10m',
                 mode='xiaorong2-1', 
                 land_use_level_num = 2,
                 **kwargs):
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
        
        # =====================================================================
        self.mode = mode
        self.land_use_level_num = land_use_level_num # L1植被 L2耕地 
        if self.mode == 'xiaorong2-1':
            pass
        elif self.mode == 'xiaorong2-2-2':
            self.new_task_weight = nn.Parameter(torch.ones(1), requires_grad=True) # 1*1   # L1的权重
            self.land_use_L1_weight = nn.Parameter(torch.ones(1), requires_grad=True) # 1*1   # L1的权重
            self.land_use_L2_weight = nn.Parameter(torch.ones(1), requires_grad=True) # 1*1   # L2的权重

        elif self.mode == 'xiaorong2-3-1':
            level_feats_channel = (self.land_use_level_num+1) * self.channels * len(self.in_channels) # (2+1)* 768*4 / 3* 768*4
            self.merge_feats = nn.Sequential(
                    nn.Conv2d(in_channels=level_feats_channel, 
                              out_channels=len(self.in_channels) * self.channels,
                              kernel_size=1,   # 1x1 卷积保持空间尺寸
                              stride=1,        # 步长为1不改变分辨率
                              padding=0  
                              ),
                    nn.BatchNorm2d(len(self.in_channels) * self.channels),
                    nn.ReLU(inplace=True)
            )
        elif self.mode == 'xiaorong2-3-2':
            level_feats_channel = (self.land_use_level_num+1) * self.channels * len(self.in_channels) # (2+1)* 768*4 / 3* 768*4
            self.merge_feats = nn.Sequential(
                    nn.Conv2d(in_channels=level_feats_channel, 
                              out_channels=len(self.in_channels) * self.channels,
                              kernel_size=1,   # 1x1 卷积保持空间尺寸
                              stride=1,        # 步长为1不改变分辨率
                              padding=0  
                              ),
                    nn.GroupNorm(num_groups=32,  # 通常设置为 32 或通道数的因子
                                num_channels=len(self.in_channels) * self.channels),
                    nn.ReLU(inplace=True)
            )
        elif self.mode == 'xiaorong2-3-3':
            self.new_task_weight = nn.Parameter(torch.ones(1), requires_grad=True) # 1*1   # L1的权重
            self.land_use_L1_weight = nn.Parameter(torch.ones(1), requires_grad=True) # 1*1   # L1的权重
            self.land_use_L2_weight = nn.Parameter(torch.ones(1), requires_grad=True) # 1*1   # L2的权重
            level_feats_channel = (self.land_use_level_num+1) * self.channels * len(self.in_channels) # (2+1)* 768*4 / 3* 768*4
            self.merge_feats = nn.Sequential(
                    nn.Conv2d(in_channels=level_feats_channel, 
                              out_channels=len(self.in_channels) * self.channels,
                              kernel_size=1,   # 1x1 卷积保持空间尺寸
                              stride=1,        # 步长为1不改变分辨率
                              padding=0  
                              ),
                    nn.BatchNorm2d(len(self.in_channels) * self.channels),
                    nn.ReLU(inplace=True)
            )


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

        if isinstance(inputs, tuple):    #tuple(Level_softmask_list, inputs) inputs:四个尺度的特征
            Level_softmask_list = inputs[0]  # [2,128,128] 需要扩展成[2,1,128,128]才能广播把
            inputs = inputs[1]
        elif isinstance(inputs, list) and len(inputs) == 4:  # backbone输出的四个尺度的特征list
            inputs = inputs
        else:
            raise TypeError('inputs must be list or tuple of Tensors')
        
        # https://blog.csdn.net/yumaomi/article/details/125376320
        inputs = self._transform_inputs(inputs)  # [2, 128, 128, 128] [2, 256, 64, 64] [2, 512, 32, 32] [2, 1024, 16, 16]

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

        
        # 可将feat和farmland_mask在通道维拼接后送入self.att_gate卷积生成注意力权重，再sigmoid后作为门控系数，再乘回feat。
        # 细分类预测：将门控后的特征gated_feat送入细分类卷积self.crop_conv，得到作物类别的logits crop_logit（形状[N, num_crops, H, W]）。
        # 最后返回两个输出：farmland_logit和crop_logit。整个forward过程将在自定义解码头类中实现。例如：


        # build top-down path  # 也就这里去做文章吧？
        # import pdb;pdb.set_trace()
        # 上采样后一层特征，并与当前层做融合
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
        fpn_outs.append(laterals[-1])  # 在这里才，全部变成了128
        # fpn_outs: [[2,768,128,128],[2,768,64,64],[2,768,32,32],[2,1024,16,16]]
        # upsample to the same size
        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        
        # fpn_outs: [[2,768,128,128],[2,768,128,128],[2,768,128,128],[2,768,128,128]]    # 在这里，对多级的特征去实施注意力啊！！！！
        fpn_outs = torch.cat(fpn_outs, dim=1) # [2,768*4,128,128]

        # ===========================================================  ATL的修改
        for i in range(len(Level_softmask_list)):
            Level_softmask_list[i] = Level_softmask_list[i].unsqueeze(1) # [2,1,128,128]  # 扩展成[2,1,128,128]才能广播把
        
        if self.mode == 'xiaorong2-1': 
            # import pdb;pdb.set_trace()
            # 消融1：将mask和cropland元素相乘，控制特征
            fpn_outs = fpn_outs + Level_softmask_list[1] * fpn_outs # 最好再来一个可学习的参数，都要一下原始特征，别全抑制了
        # elif self.mode == 'xiaorong2-2':
        #     # 消融2：通过注意力的形式，来提高对特定区域的关注。类似于那个Hiera-Unet。一级的输出*L1的mask  
        #     # 再合并到特征图上，去输出二级的，在合并到特征图上去输出三级的。或者就类似于本身的Hiera。
        #     crop_land_seglogit_softmax = Level_softmask_list[1]
        #     att = torch.sigmoid(self.att_gate(torch.cat([fpn_outs, crop_land_seglogit_softmax], dim=1))) # 这将在门控中引入可学习参数
        #     gated_feat = fpn_outs * att  # 这将在门控中引入可学习参数，使模型自行调节对mask的依赖程度。
        elif self.mode == 'xiaorong2-2':
            # import pdb;pdb.set_trace()
            # 消融3，在特征提取上面，逐步的去增强相关区域特征的关注度
            # 如，我关注的是飞机，L1是人造地表 L2是交通设施的区域 L3是机场的区域 L4是飞机。
            fpn_outs = fpn_outs + Level_softmask_list[0]*fpn_outs + Level_softmask_list[1]*fpn_outs # 植被mask*特征 + 耕地mask*特征 抑制了这些地方的特征，突出了1*植被区和2*耕地区 
        elif self.mode == 'xiaorong2-2-2':
            fpn_outs = fpn_outs*self.new_task_weight +\
                       Level_softmask_list[0]*fpn_outs*self.land_use_L1_weight +\
                       Level_softmask_list[1]*fpn_outs*self.land_use_L2_weight # 植被mask*特征 + 耕地mask*特征 抑制了这些地方的特征，突出了1*植被区和2*耕地区 

        elif self.mode == 'xiaorong2-3':
            # import pdb;pdb.set_trace()
            # 消融3，在特征提取上面，逐步的去增强相关区域特征的关注度
            # 如，我关注的是飞机，L1是人造地表 L2是交通设施的区域 L3是机场的区域 L4是飞机。
            # 植被mask*特征 + 耕地mask*特征 抑制了这些地方的特征，突出了1*植被区和2*耕地区 
            fpn_outs = torch.cat([fpn_outs, 
                                  Level_softmask_list[0]*fpn_outs, 
                                  Level_softmask_list[1]*fpn_outs], dim=1)  # [2,768*4,128,128], [2,768*4,128,128] --> [2,768*12,128,128]
            fpn_outs = self.merge_feats(fpn_outs) # [2,768*8,128,128] --> [2,768*4,128,128]
        
        elif self.mode == 'xiaorong2-2-2':
            fpn_outs = torch.cat([fpn_outs*self.new_task_weight, 
                                  Level_softmask_list[0]*fpn_outs*self.land_use_L1_weight, 
                                  Level_softmask_list[1]*fpn_outs*self.land_use_L2_weight], dim=1)  # [2,768*4,128,128], [2,768*4,128,128] --> [2,768*12,128,128]
            fpn_outs = self.merge_feats(fpn_outs) # [2,768*8,128,128] --> [2,768*4,128,128]

        # fpn_outs: [2,3072,128,128]
        feats = self.fpn_bottleneck(fpn_outs)  # [2,3072,128,128] --> [2,768,128,128]  #用抑制或者增强后的特征图，再去实施精细作物类别的提取？
        # ConvModule(
        # (conv): Conv2d(4096, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        # (bn): _BatchNormXd(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # (activate): ReLU(inplace=True)
        # )
        # feats.shape: torch.Size([2, 1024, 128, 128])
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)  # [2,768,128,128]
        output = self.cls_seg(output)  # [2,4,128,128]
        return output
