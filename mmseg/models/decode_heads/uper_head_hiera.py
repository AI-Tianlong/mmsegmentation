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

from mmseg.models.losses.atl_hiera_37_loss_convseg import (convert_low_level_label_to_High_level,
                                                           L1_L2map,L2_L3map,
                                                           FiveBillion_18Classes_HieraMap_nobackground)


class ProjectionHead(nn.Module):
    """ProjectionHead, project feature map to specific channels.

    Args:
        dim_in (int): Input channels.
        norm_cfg (dict): config of norm layer.
        proj_dim (int): Output channels. Default: 256.
        proj (str): Projection type, 'linear' or 'convmlp'. Default: 'convmlp'
    """

    def __init__(self,
                 dim_in: int,
                 norm_cfg: dict,
                 proj_dim: int = 256,
                 proj: str = 'convmlp'):
        super().__init__()
        assert proj in ['convmlp', 'linear']
        if proj == 'linear':  # 投影，线性的话，就用1x1的卷积，将输入通道数变为输出通道数
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        elif proj == 'convmlp':  # 如果是convmlp，就用两个1x1的卷积，中间加上BN和ReLU
            self.proj = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=1),
                build_norm_layer(norm_cfg, dim_in)[1], nn.ReLU(inplace=True),
                nn.Conv2d(dim_in, proj_dim, kernel_size=1))

    def forward(self, x):
        return torch.nn.functional.normalize(self.proj(x), p=2, dim=1)



@MODELS.register_module()
class UPerHead_Hiera(BaseDecodeHead):
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

                 num_classes_level_list: List[int] = [4,9,18],    # 层级的类别
                 results_merge_hiera: bool = True,   # 输出结果，融合hiera的输出
                 hiera_mode:str = 'xiaorong1',       # 用来修改消融实验的结构的
                 proj: str = 'convmlp',              # 特征投影成embedding          
                 
                 **kwargs):
        
        # PSP Module
        num_classes = num_classes_level_list[-1]  # 【ATL-LOG】去创建 self.conv_seg
        super().__init__(num_classes=num_classes, # 【ATL-LOG】去创建 self.conv_seg
                         input_transform='multiple_select',
                         **kwargs)
        
        #============= 创建Hiera需要用到的模块。=======================
        self.results_merge_hiera = results_merge_hiera
        self.hiera_mode = hiera_mode
        if isinstance(num_classes_level_list, list):
            self.num_classes_level_list = num_classes_level_list  # [5,9,10]


            if self.hiera_mode == 'xiaorong1':
                # 消融实验1  # 多层级间没有交互。只是输出三个通道，最后融合输出。
                self.conv_seg_L1 = nn.Conv2d(self.channels, num_classes_level_list[0], kernel_size=1) #(1024-->5)
                self.conv_seg_L2 = nn.Conv2d(self.channels, num_classes_level_list[1], kernel_size=1)
                self.conv_seg_L3 = nn.Conv2d(self.channels, num_classes_level_list[2], kernel_size=1)
                self.conv_seg #(1024-->12)
            elif self.hiera_mode == 'xiaorong2':
                # 消融实验2  # 多层级间假交互，只是把第一个的特征图叠加在第二个上，最后融合输出。
                self.conv_seg_L1 = nn.Conv2d(self.channels,                
                                             num_classes_level_list[0], 
                                             kernel_size=1)  #(1024-->4)      
                self.conv_seg_L2 = nn.Conv2d(self.channels+num_classes_level_list[0], 
                                             num_classes_level_list[1], 
                                             kernel_size=1)  #(1024+4-->10)      
                self.conv_seg_L3 = nn.Conv2d(self.channels+num_classes_level_list[0]+num_classes_level_list[1], 
                                             num_classes_level_list[2], 
                                             kernel_size=1)  #(1024+4+10-->18)   
                # self.conv_seg # 就不要了, 但是会默认创建一个，无所谓啦

            elif self.hiera_mode == 'xiaorong3':
                # 消融实验3 # 多层级间有空间注意力的交互，最后融合输出。这个空间注意力可以换一个。
                self.conv_seg_L1 = nn.Conv2d(self.channels, num_classes_level_list[0], kernel_size=1) #(1024-->5)
                self.conv_attention_L1 = nn.Conv2d(num_classes_level_list[0], 1, kernel_size=1) #(5-->1)
                
                self.conv_seg_L2 = nn.Conv2d(self.channels, num_classes_level_list[1], kernel_size=1)
                self.conv_attention_L2 = nn.Conv2d(num_classes_level_list[1], 1, kernel_size=1) #(10-->1)
                
                self.conv_seg_L3 = self.conv_seg #(1024-->12)

            elif self.hiera_mode == 'xiaorong4':
                self.sigmoid=nn.Sigmoid()
                
                # L1
                self.conv_seg_L1 = nn.Conv2d(self.channels, num_classes_level_list[0], kernel_size=1) #[2,1024,128,128]->[2,4,128,128]
                # L1 channel_attentation
                self.max_pool_L1 = nn.AdaptiveMaxPool2d(output_size=1) # [2,4,128,128]-->[2,4,1,1]  
                self.avg_pool_L1 = nn.AdaptiveAvgPool2d(output_size=1) # [2,4,128,128]-->[2,4,1,1]  
                self.mlp_L1=nn.Sequential(
                    nn.Linear(in_features=num_classes_level_list[0],out_features=self.channels,bias=False), # [2,4,1,1]-->[2,1024,1,1]
                    nn.ReLU())
                # L1 spatial_attentation
                self.conv_L1=nn.Conv2d(in_channels=2, out_channels=1, 
                    kernel_size=7 ,
                    stride=1,
                    padding=7//2,bias=False)
                
                # L2
                self.conv_seg_L2 = nn.Conv2d(self.channels, num_classes_level_list[1], kernel_size=1)
                # L2 channel_attentation
                self.max_pool_L2 = nn.AdaptiveMaxPool2d(output_size=1) # [2,9,128,128]-->[2,9,1,1]  
                self.avg_pool_L2 = nn.AdaptiveAvgPool2d(output_size=1) # [2,9,128,128]-->[2,9,1,1]  
                self.mlp_L2=nn.Sequential(
                    nn.Linear(in_features=num_classes_level_list[1],out_features=self.channels,bias=False), # [2,4,1,1]-->[2,1024,1,1]
                    nn.ReLU())
                # L2 spatial_attentation
                self.conv_L2=nn.Conv2d(in_channels=2, out_channels=1, 
                    kernel_size=7 ,
                    stride=1,
                    padding=7//2,bias=False)

                self.conv_seg_L3 = self.conv_seg #(1024-->12)


            else:
                raise ValueError(f'不支持的 hiera_mode: {self.hiera_mode}, 请检查消融实验配置')

        self.proj_head = ProjectionHead(dim_in=self.in_channels[-1],   # backbone的最后一个特征图的维度
                                        norm_cfg=self.norm_cfg, 
                                        proj=proj)
        self.register_buffer('step', torch.zeros(1))
        # ====================== Hiera ======================


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
        
        import pdb;pdb.set_trace
        embedding = self.proj_head(inputs[-1])  # For TreeTriplet Loss

        if self.hiera_mode == 'xiaorong1':
            output_L1 = self.cls_seg_hiear(decode_head_outputs, self.conv_seg_L1) # [2,1024,128,128]->[2,4,128,128]
            output_L2 = self.cls_seg_hiear(decode_head_outputs, self.conv_seg_L2) # [2,1024,128,128]->[2,9,128,128]
            output_L3 = self.cls_seg_hiear(decode_head_outputs, self.conv_seg_L3) # [2,1024,128,128]->[2,18,128,128]
            output_list = [output_L1, output_L2, output_L3]
            
            self.step += 1
            return output_list, embedding

        elif self.hiera_mode == 'xiaorong2':
            output_L1 = self.cls_seg_hiear(decode_head_outputs, self.conv_seg_L1) # [2,1024,128,128]->[2,4,128,128]
            output_L2 = self.cls_seg_hiear(decode_head_outputs, self.conv_seg_L2) # [2,1024+4,128,128]->[2,9,128,128]
            output_L3 = self.cls_seg_hiear(decode_head_outputs, self.conv_seg_L3) # [2,1024+4+9,128,128]->[2,18,128,128]
            output_list = [output_L1, output_L2, output_L3]
            
            self.step += 1
            return output_list, embedding
        
        elif self.hiera_mode == 'xiaorong3':
            
            output_L1 = self.cls_seg_hiear(decode_head_outputs, self.conv_seg_L1)  # [2,1024,128,128]->[2,4,128,128]
            output_L1_attn = self.conv_attention_L1(output_L1)                     # [2,4,128,128]->[2,1,128,128]
            decode_head_outputs = decode_head_outputs + output_L1_attn # 加还是×啊？
            
            output_L2 = self.cls_seg_hiear(decode_head_outputs, self.conv_seg_L2)  # [2,1024,128,128]->[2,9,128,128]
            output_L2_attn = self.conv_attention_L2(output_L2)                     # [2,9,128,128]->[2,1,128,128]
            decode_head_outputs = decode_head_outputs + output_L2_attn    

            output_L3 = self.cls_seg_hiear(decode_head_outputs, self.conv_seg_L3)  # [2,1024,128,128]->[2,18,128,128]
            output_list = [output_L1, output_L2, output_L3]
            
            self.step += 1
            return output_list, embedding

        elif self.hiera_mode == 'xiaorong4':
            # import pdb; pdb.set_trace()
            output_L1 = self.cls_seg_hiear(decode_head_outputs, self.conv_seg_L1)  # [2,1024,128,128]->[2,4,128,128] 
            maxout_L1 = self.max_pool_L1(output_L1) # [2,4,128,128]-->[2,4,1,1]
            maxout_L1 = self.mlp_L1(maxout_L1.view(maxout_L1.size(0),-1))  # [2,1024]
            avgout_L1 = self.avg_pool_L1(output_L1) # [2,4,128,128]-->[2,4,1,1]
            avgout_L1 = self.mlp_L1(avgout_L1.view(avgout_L1.size(0),-1)) # [2,4,1,1]-->[2,768]
            channel_out_L1 = self.sigmoid(maxout_L1+avgout_L1) # [2,1024]
            channel_out_L1 = channel_out_L1.view(decode_head_outputs.size(0),decode_head_outputs.size(1),1,1) # [2, 1024,1,1] 
            channel_out_L1 = channel_out_L1*decode_head_outputs  #广播机制 # [2, 1024,128,128] 
            max_out_L1,_ = torch.max(output_L1,dim=1,keepdim=True) # [2,1,128,128]
            mean_out_L1 = torch.mean(output_L1,dim=1,keepdim=True) # [2,1,128,128]
            spatial_out_L1 = torch.cat((max_out_L1,mean_out_L1),dim=1) #[2,2,128,128]
            spatial_out_L1 = self.sigmoid(self.conv_L1(spatial_out_L1)) #[2,1,128,128]
            decode_head_outputs=spatial_out_L1*channel_out_L1 # 然后再乘上系数。[2,1,128,128]*[2,1024,128,128]


            output_L2 = self.cls_seg_hiear(decode_head_outputs, self.conv_seg_L2)  # [2,1024,128,128]->[2,9,128,128] 
            maxout_L2 = self.max_pool_L2(output_L2) # [2,9,128,128]-->[2,9,1,1]
            maxout_L2 = self.mlp_L2(maxout_L2.view(maxout_L2.size(0),-1))  #[2,1024]
            avgout_L2 = self.avg_pool_L2(output_L2) # [2,4,128,128]-->[2,9,1,1]
            avgout_L2 = self.mlp_L2(avgout_L2.view(avgout_L2.size(0),-1)) # [2,4,1,1]-->[2,1024,1,1]
            channel_out_L2 = self.sigmoid(maxout_L2+avgout_L2) # [2,1024]
            channel_out_L2 = channel_out_L2.view(decode_head_outputs.size(0),decode_head_outputs.size(1),1,1) # [2, 1024,1,1] 
            channel_out_L2 = channel_out_L2*decode_head_outputs  #广播机制
            max_out_L2,_ = torch.max(output_L2,dim=1,keepdim=True)
            mean_out_L2 = torch.mean(output_L2,dim=1,keepdim=True) 
            spatial_out_L2 = torch.cat((max_out_L2,mean_out_L2),dim=1)
            spatial_out_L2 = self.sigmoid(self.conv_L2(spatial_out_L2))
            decode_head_outputs=spatial_out_L2*channel_out_L2 # 然后再乘上系数。
        
            output_L3 = self.cls_seg_hiear(decode_head_outputs, self.conv_seg_L3)
            output_list = [output_L1, output_L2, output_L3]
            self.step += 1
            
            return output_list, embedding
        else:
            raise ValueError(f'不支持的 hiera_mode: {self.hiera_mode}, 请检查消融实验配置')

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)  # [2,1024,128,128]
        output_list, embedding = self.hiera_module(inputs, output)  # [2,65,128,128]

        return output_list, embedding
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
        if isinstance(seg_logits, tuple):
            if len(seg_logits) == 2:
                seg_logits, embedding = seg_logits

                if isinstance(seg_logits, list) and len(seg_logits) == 3:
                    seg_logits = seg_logits
                elif isinstance(seg_logits, torch.Tensor) and seg_logits.shape[1]==sum(self.num_classes_level_list):
                    seg_logits = seg_logits
        else:
            raise TypeError(f'UperHead_Hiera的输出应该是一个tuple, 包含 `output_list` 和 `embedding`.',
                            f'但得到type(seg_logits)')

        # seg_logits: [2,65,128,128]
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
                    self.step,  # for TreeTriplet loss
                    embedding,  # for TreeTriplet loss
                    seg_logits, # output_list[L1,L2,L3]
                    seg_label,  # L3级别的label
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
                # print(loss)
                # import pdb; pdb.set_trace()
            else:
                # pdb.set_trace()
                loss[loss_decode.loss_name] += loss_decode(
                    self.step,
                    embedding,
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
        seg_label_list = convert_low_level_label_to_High_level(seg_label, FiveBillion_18Classes_HieraMap_nobackground)
        loss['acc_seg_L1'] = accuracy(seg_logits_L1, seg_label_list[0], ignore_index=self.ignore_index)
        loss['acc_seg_L2'] = accuracy(seg_logits_L2, seg_label_list[1], ignore_index=self.ignore_index)
        loss['acc_seg_L3'] = accuracy(seg_logits_L3, seg_label_list[2], ignore_index=self.ignore_index)
        loss['acc_seg'] = accuracy(seg_logits_L3, seg_label, ignore_index=self.ignore_index)

        if self.results_merge_hiera:
            seg_logits_merge = self.merge_hiera_results(seg_logits)
            # seg_logits = seg_logits[2] # 输出融合后L3的特征图, 去计算精度
            seg_logits_L1_merge = seg_logits_merge[0]
            seg_logits_L2_merge = seg_logits_merge[1]
            seg_logits_L3_merge = seg_logits_merge[2]

            loss['acc_seg_merge_L1'] = accuracy(seg_logits_L1_merge, seg_label_list[0], ignore_index=self.ignore_index)
            loss['acc_seg_merge_L2'] = accuracy(seg_logits_L2_merge, seg_label_list[1], ignore_index=self.ignore_index)
            loss['acc_seg_merge_L3'] = accuracy(seg_logits_L3_merge, seg_label_list[2], ignore_index=self.ignore_index)
            loss['acc_seg_merge'] = accuracy(seg_logits_L3_merge, seg_label, ignore_index=self.ignore_index)

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
        if isinstance(seg_logits, tuple):
            if len(seg_logits) == 2:
                seg_logits, embedding = seg_logits  #推理只需要 seg_logits

                # 融合 L1+L2+L3 三层
                if isinstance(seg_logits, list) and len(seg_logits) == 3:
                    # 合并L1 L2 L3 级的推理结果
                    if self.results_merge_hiera:
                        seg_logits = self.merge_hiera_results(seg_logits)
                        seg_logits = seg_logits[2] # 仅输出融合后L3的特征图
                    else:   
                        seg_logits = seg_logits[2] # 直接输出L3的特征图
            else:
                raise TypeError(f'seg_logits 应该是个 tuple',f'但是得到了个 {type(seg_logits)}')

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
        
        return seg_logits
    

    def merge_hiera_results(self, seg_logits:List[Tensor], sigmoid:bool=True) -> Tensor:
        """Merge L1 L2 L3 level results.
        
        Args: 
            seg_logits (List[Tensor]): The output from decode head forward function.
            sigmoid (bool): Whether to use sigmoid to normalize the logits. Default: False.
        
        Returns:
            Tensor: Final segmentation logits map.
        """
        # from mmseg.models.losses.atl_hiera_37_loss_convseg import L1_L2map,L2_L3map
        # L1_L2map = [[0,1,2], [3], [4,5,6,7],[8]]  # L2和L1的层级关系，数字代表L2中的类别
        # L2_L3map = [[0,1],[2], [3,4],[5,6,7],[8],[9,10],[11,12],[13,14,15,16],[17],[18]] # L3和L2的层级关系，数字代表L3中的类别

        # 这里加之前，需要用sigmoid 归一化一下，再去加嘛？ 试一下
        if sigmoid:
            for index in range(len(seg_logits)):
                seg_logits[index] = F.sigmoid(seg_logits[index])
        
        seg_logits_L1 = seg_logits[0]  # [2,4,640,640]
        seg_logits_L2 = seg_logits[1]  # [2,9,640,640]
        seg_logits_L3 = seg_logits[2]  # [2,18,640,640]

        seg_logits_merge_L1 = seg_logits_L1.clone()
        seg_logits_merge_L2 = seg_logits_L2.clone()
        seg_logits_merge_L3 = seg_logits_L3.clone()

        # 融合 L1、L2到 L2 , 这里，其实也可以给加每个特征图加一个可学习的权重。
        for L1_index in range(len(L1_L2map)):     # L1 的 0 1 2 3
            L2_index_group = L1_L2map[L1_index] 
            for L2_index in L2_index_group:      # L2 的 0,1,2 
                # print(f'L1[{L1_index}] + L2[{L2_index}]')
                L1_seg_logit = seg_logits_L1[:, L1_index, :, :] # [2, 640, 640], 还是得四维的啊
                L2_seg_logit = seg_logits_L2[:, L2_index, :, :]

                seg_logits_merge_L2[:,L2_index,:,:] = L1_seg_logit + L2_seg_logit

        # 融合 L1、L2、L3 到L3
        for L1_index in range(len(L1_L2map)):     # L1 的 0 1 2 3
            L2_index_group = L1_L2map[L1_index] 
            for L2_index in L2_index_group:      # L2 的 0,1,2 
                L3_index_group = L2_L3map[L2_index]
                for L3_index in L3_index_group:  # L3 的 0 1 
                    # print(f'L1[{L1_index}] + L2[{L2_index}] + L3[{L3_index}]')
                    L1_seg_logit = seg_logits_L1[:, L1_index, :, :] # [2, 640, 640], 还是得四维的啊
                    L2_seg_logit = seg_logits_L2[:, L2_index, :, :]
                    L3_seg_logit = seg_logits_L3[:, L3_index, :, :]

                    seg_logits_merge_L3[:,L3_index,:,:] = L1_seg_logit + L2_seg_logit + L3_seg_logit

        return [seg_logits_merge_L1, seg_logits_merge_L2, seg_logits_merge_L3]

