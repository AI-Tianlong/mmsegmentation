# Copyright (c) OpenMMLab. All rights reserved.
from venv import logger

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from torch import Tensor
from mmengine.device import get_device
from mmseg.registry import MODELS
from mmseg.utils import SampleList
from ..utils import resize
from .decode_head import BaseDecodeHead
from .psp_head import PPM

import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule, build_norm_layer
from mmseg.models.losses import accuracy
from mmseg.utils import SampleList, ConfigType
from typing import List, Tuple


from mmseg.models.losses.hcc_loss import (convert_low_level_label_to_High_level,
                                          L1_L2map, L2_L3map,
                                          MM_5B_18_hiera_structure)

class HSM_MergeBlock(nn.Module):
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
class UPerHead_HSM(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.
    This head is the implementation of `UPerNet <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
        
        num_classes_level_list(List[int]): Hiera classes_num list, default:[4,9,18].
        results_merge_hiera (bool): 最终的特征图输出,是否融合L1 L2 L3, default: True.
        hiera_mode (str): hiera 模块的mode，用来消融调试, 'xiaorong1', 'xiaorong2'...
    
    Returns:
        output_list, embedding
    """

    def __init__(self, 
                 pool_scales=(1, 2, 3, 6), 
                 ouput_level: str = 'L3',  # 推理时输出的层级，训练时该参数无效
                 num_classes_level_list: List[int] = [4,9,18],    # 层级的类别
                 path_merge: bool = True,   # 输出结果，融合hiera的输出
                 hiera_mode:str = 'xiaorong1',       # 用来修改消融实验的结构的
                 **kwargs):
        
        # PSP Module
        num_classes = num_classes_level_list[-1]  # 【ATL-LOG】去创建 self.conv_seg
        super().__init__(num_classes=num_classes, # 【ATL-LOG】去创建 self.conv_seg
                         input_transform='multiple_select',
                         **kwargs)
        
        #============= 创建Hiera需要用到的模块。=======================
        self.test_output_level = ouput_level  # 测试推理时输出的层级
        self.results_path_merge = path_merge
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
                self.stage1_1to2_MB = HSM_MergeBlock(num_classes_level_list[0], num_classes_level_list[1])# For L2
                self.stage1_w12 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
                self.stage1_w22 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
                
                self.stage1_1to3_MB = HSM_MergeBlock(num_classes_level_list[0], num_classes_level_list[2])# For L3
                self.stage1_2to3_MB = HSM_MergeBlock(num_classes_level_list[1], num_classes_level_list[2])
                self.stage1_w13 = nn.Parameter(torch.tensor(1.0), requires_grad=True) 
                self.stage1_w23 = nn.Parameter(torch.tensor(1.0), requires_grad=True) 
                self.stage1_w33 = nn.Parameter(torch.tensor(1.0), requires_grad=True) 
                

            elif self.hiera_mode == 'xiaorong4':
               # 消融实验4  # 多层级间只有从细到粗交互，然后独立计算CELoss。
                self.conv_seg_L1 = nn.Conv2d(self.channels, num_classes_level_list[0], kernel_size=1) #(1024-->5)
                self.conv_seg_L2 = nn.Conv2d(self.channels, num_classes_level_list[1], kernel_size=1)
                self.conv_seg_L3 = self.conv_seg

                # stage2: fine to coarse 18-->9 | 18-->4  9-->4
                self.stage2_3to2_MB = HSM_MergeBlock(num_classes_level_list[2], num_classes_level_list[1])# For L2
                self.stage2_y32 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
                self.stage2_y22 = nn.Parameter(torch.tensor(1.0), requires_grad=True)

                self.stage2_3to1_MB = HSM_MergeBlock(num_classes_level_list[2], num_classes_level_list[0])# For L1
                self.stage2_2to1_MB = HSM_MergeBlock(num_classes_level_list[1], num_classes_level_list[0])# For L1
                self.stage2_y31 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
                self.stage2_y21 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
                self.stage2_y11 = nn.Parameter(torch.tensor(1.0), requires_grad=True)

            elif self.hiera_mode == 'xiaorong5':
                # 消融实验5  # 多层级间双向交互，然后独立计算CELoss/用HCC。
                self.conv_seg_L1 = nn.Conv2d(self.channels, num_classes_level_list[0], kernel_size=1) #[2,1024,128,128]->[2,4,128,128]
                self.conv_seg_L2 = nn.Conv2d(self.channels, num_classes_level_list[1], kernel_size=1) #(1024-->9)
                self.conv_seg_L3 = self.conv_seg #(1024-->18)
                
                # stage1: coarse to fine 4-->9 | 4->18 9->18
                self.stage1_1to2_MB = HSM_MergeBlock(num_classes_level_list[0], num_classes_level_list[1])# For L2
                self.stage1_w12 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
                self.stage1_w22 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
                
                self.stage1_1to3_MB = HSM_MergeBlock(num_classes_level_list[0], num_classes_level_list[2])# For L3
                self.stage1_2to3_MB = HSM_MergeBlock(num_classes_level_list[1], num_classes_level_list[2])
                self.stage1_w13 = nn.Parameter(torch.tensor(1.0), requires_grad=True) 
                self.stage1_w23 = nn.Parameter(torch.tensor(1.0), requires_grad=True) 
                self.stage1_w33 = nn.Parameter(torch.tensor(1.0), requires_grad=True) 
                
                # stage2: fine to coarse 18-->9 | 18-->4  9-->4
                self.stage2_3to2_MB = HSM_MergeBlock(num_classes_level_list[2], num_classes_level_list[1])# For L2
                self.stage2_y32 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
                self.stage2_y22 = nn.Parameter(torch.tensor(1.0), requires_grad=True)

                self.stage2_3to1_MB = HSM_MergeBlock(num_classes_level_list[2], num_classes_level_list[0])# For L1
                self.stage2_2to1_MB = HSM_MergeBlock(num_classes_level_list[1], num_classes_level_list[0])# For L1
                self.stage2_y31 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
                self.stage2_y21 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
                self.stage2_y11 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
                
            else:
                raise ValueError(f'不支持的 hiera_mode: {self.hiera_mode}, 请检查消融实验配置')
            

        # ================== 原始UPerHead的结构 =========
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
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
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

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
        inputs = self._transform_inputs(inputs)

        # build laterals,
        # 3xConvModule{Conv2d(1024, 1024, (1,1), (1,1), bias=False) + bn + ReLU}
        #
        # laterals = [[2, 1024, 128, 128], [2, 1024, 64, 64], [2, 1024, 32, 32]]
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # laterals = [[2, 1024, 128, 128], [2, 1024, 64, 64], [2, 1024, 32, 32], [2, 1024, 16, 16]]
        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)
        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]

        # append psp feature
        fpn_outs.append(laterals[-1])
        # fpn_outs: [[2,1024,128,128],[2,1024,64,64],[2,1024,32,32],[2,1024,16,16]]

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        # fpn_outs: [[2,1024,128,128],[2,1024,128,128],[2,1024,128,128],[2,1024,128,128]]
        fpn_outs = torch.cat(fpn_outs, dim=1)
        # fpn_outs: [2,4096,128,128]

        feats = self.fpn_bottleneck(fpn_outs)
        # ConvModule(
        # (conv): Conv2d(4096, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        # (bn): _BatchNormXd(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # (activate): ReLU(inplace=True)
        # )
        # feats.shape: torch.Size([2, 1024, 128, 128])
        return feats



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
            # 消融实验5  # 多层级间双向交互，然后独立计算CELoss/用HCC。
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
            
            output_list = [L1_out, L2_out, L3_out]
            return output_list
        
        else:
            raise ValueError(f'不支持的 hiera_mode: {self.hiera_mode}, 请检查消融实验配置')

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)  # [2,1024,128,128]
        output_list = self.hiera_module(inputs, output)  # [2,65,128,128]

        return output_list
        # output = self.cls_seg(output)  # [2,65,128,128]
        # return output


    # =================== Hiera 修改 LOSS 和 predict 方式 ===============
    
    def loss_by_feat(self, 
                     seg_logits: Tensor,
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
            seg_logits_L1 = seg_logits[0] # 直接输出L3的特征图
            seg_logits_L2 = seg_logits[1]
            seg_logits_L3 = seg_logits[2]
        else:
            raise TypeError(f'seg_logits 应该是个 list ',f'但是得到了个 {type(seg_logits)}')

        # L1、L2、L3级别的精度
        seg_label_list = convert_low_level_label_to_High_level(seg_label, MM_5B_18_hiera_structure)
        loss['acc_seg_L1'] = accuracy(seg_logits_L1, seg_label_list[0], ignore_index=self.ignore_index)
        loss['acc_seg_L2'] = accuracy(seg_logits_L2, seg_label_list[1], ignore_index=self.ignore_index)
        loss['acc_seg_L3'] = accuracy(seg_logits_L3, seg_label_list[2], ignore_index=self.ignore_index)
        loss['acc_seg'] = accuracy(seg_logits_L3, seg_label, ignore_index=self.ignore_index)

        if self.results_path_merge:
            path_merge_mask = self.results_path_merge_func(seg_logits) # 选择最优路径的mask，获得的是mask，而不是seglogits？
            # seg_logits = seg_logits[2] # 输出融合后L3的特征图, 去计算精度

            path_merge_mask_L1 = path_merge_mask[:,0:1,:,:]
            path_merge_mask_L2 = path_merge_mask[:,1:2,:,:]
            path_merge_mask_L3 = path_merge_mask[:,2:3,:,:]

            loss['acc_seg_path_merge_L1'] = accuracy_path_merge_results(path_merge_mask_L1, seg_label_list[0], ignore_index=self.ignore_index)
            loss['acc_seg_path_merge_L2'] = accuracy_path_merge_results(path_merge_mask_L2, seg_label_list[1], ignore_index=self.ignore_index)
            loss['acc_seg_path_merge_L3'] = accuracy_path_merge_results(path_merge_mask_L3, seg_label_list[2], ignore_index=self.ignore_index)
            
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


        if isinstance(batch_img_metas[0]['img_shape'], torch.Size):
            # slide inference
            size = batch_img_metas[0]['img_shape']
        elif 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape'][:2]
        else:
            size = batch_img_metas[0]['img_shape']
            
        seg_logits = resize(
            input=seg_logits,
            size=size,
            mode='bilinear',
            align_corners=self.align_corners)
        
       
        if self.results_path_merge:
            pred_mask = self.results_path_merge_func(seg_logits) #直接是最后的mask啊

            return [seg_logits, pred_mask]
        # 这里是不是应该写在后处理里啊？  写在这里好像不太对，应为post要的是seglogits然后处理。
        return seg_logits
    

    def results_path_merge_func(self, seg_logits:List[Tensor], sigmoid:bool=True) -> Tensor:
        """Path-based hierarchical inference via joint score maximization.
        
        Args: 
            seg_logits (List[Tensor]): The output from decode head forward function.
            sigmoid (bool): Whether to use sigmoid to normalize the logits. Default: False.
        
        Returns:
            Tensor: Final segmentation logits map.
        """

        
        # === 1. Normalize ===
        if sigmoid:
            seg_logits = [F.sigmoid(logit) for logit in seg_logits]
        
        seg_L1, seg_L2, seg_L3 = seg_logits  # shapes: [B, C1, H, W], etc.
        B, _, H, W = seg_L1.shape


        # 找出每一条合法路径的顺序和结果。
        # === 3. Generate all valid hierarchical paths ===
        path_list = []  # List of (L1_idx, L2_idx, L3_idx)
        for L1_idx, L2_group in enumerate(L1_L2map):  # 0, [0,1,2] （L1的index 和L2的index）
            for L2_idx in L2_group:  # [0,1],[2],[3,4](列表代表这些L3index是一组的)
                for L3_idx in L2_L3map[L2_idx]:
                    path_list.append((L1_idx, L2_idx, L3_idx))
        
        # [(0, 0, 0), (0, 0, 1), (0, 1, 2), (0, 2, 3), 
        # (0, 2, 4), (1, 3, 5), (1, 3, 6), (1, 3, 7), 
        # (2, 4, 8), (2, 5, 9), (2, 5, 10), (2, 6, 11), 
        # (2, 6, 12), (2, 7, 13), (2, 7, 14), (2, 7, 15),
        #  (2, 7, 16), (3, 8, 17)]

        num_paths = len(path_list) #合法路径的个数
        path_scores = torch.zeros(B, num_paths, H, W, device=seg_L1.device) #每一个像素上，每一条路径的得分
        
        # === 4. Compute joint score for each path ===
        for idx, (l1, l2, l3) in enumerate(path_list):
            score = 0.3*seg_L1[:, l1, :, :] + 0.3 * seg_L2[:, l2, :, :] + 0.4 * seg_L3[:, l3, :, :] #调节因子，平衡三个通道的权重
            path_scores[:, idx, :, :] = score

        # === 5. Argmax over all valid paths ===
        best_path_idx = torch.argmax(path_scores, dim=1)  # shape: [B, H, W], 每个像素上，最优路径
        # best_path_idx 的值是路径的索引，0的话，代表0-0-0 最优，则l1=0, l2=0, l3=0
        
        # === 6. Convert path index to hierarchical labels ===
        final_pred = torch.zeros(B, len(seg_logits), H, W, dtype=torch.long, device=seg_L1.device).fill_(255)
        for idx, (l1, l2, l3) in enumerate(path_list):
            mask = (best_path_idx == idx)  # idx = 0，代表 0-0-0 最优
            final_pred[:, 0][mask] = l1
            final_pred[:, 1][mask] = l2
            final_pred[:, 2][mask] = l3

        return final_pred  # shape: [B, 3, H, W] #最后merge后的结果



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
