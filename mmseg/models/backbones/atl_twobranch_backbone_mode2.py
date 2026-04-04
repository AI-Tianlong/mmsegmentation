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


# mmcv 1.x
# from mmdet.models.builder import BACKBONES
# from mmdet.utils import get_root_logger

@MODELS.register_module()
# class TwoBranch_backbone_mode2(nn.Module):
class TwoBranch_backbone_mode2(BaseModule):
    def __init__(self,
                branch1_backbone={},
                branch2_backbone={},
                
                n_points=4,
                deform_num_heads=6,
                with_cffn=False, 
                cffn_ratio=0.25,
                deform_ratio=1.0,
                is_dino=False,
                interaction_proj=True,

                interact_attn_type='normal',
                interaction_drop_path_rate=0.3,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                pretrained=None,
                with_simple_fpn=True,
                out_interaction_indexes=[],
                cal_flops=False,
                ):
        
        super().__init__()

        if norm_layer == "none":
            norm_layer = nn.Identity
        
        self.interact_attn_type = interact_attn_type
        self.out_interaction_indexes = out_interaction_indexes
        self.cal_flops = cal_flops

        branch1_backbone_cfg = branch1_backbone.copy()
        branch2_backbone_cfg = branch2_backbone.copy()
    
        # 这里的交互方式，参考PIIP，并有点教师网络的意思啊？
        # 两个分支的权重
        self.w1 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.w2 = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        self.branch1_interaction_indexes = branch1_backbone_cfg.pop("interaction_indexes")
        self.branch2_interaction_indexes = branch2_backbone_cfg.pop("interaction_indexes")
        self.branch1_real_size = branch1_backbone_cfg.pop("img_real_size") # pop 从config里去掉
        self.branch2_real_size = branch2_backbone_cfg.pop("img_real_size")

        # 构建两个分支的backbone模型，并用初始化, 这里貌似并不会去主动初始化了，应该去显示的初始化
        self.branch1_backbone = MODELS.build(branch1_backbone_cfg)  # convnext
        self.branch2_backbone = MODELS.build(branch2_backbone_cfg)  # convnext
        # self.branch1_backbone land_use分支 不需要梯度
        self.branch1_backbone.eval()
        for param in self.branch1_backbone.parameters():
            param.requires_grad = False

        assert len(self.branch1_interaction_indexes) == len(self.branch2_interaction_indexes)
        num_interactions = len(self.branch1_interaction_indexes)
        
        if isinstance(self.branch1_backbone, ConvNeXt):
            dims1 = self.branch1_backbone.out_dims
        elif isinstance(self.branch1_backbone, DeiT3):
            depth = len(self.branch1_backbone.layers)
            dims1 = [self.branch1_backbone.embed_dim] * depth
            self.branch1_backbone.downsample_ratios = [self.branch1_backbone.patch_size] * depth

        if isinstance(self.branch2_backbone, ConvNeXt):
            dims2 = self.branch2_backbone.out_dims
        elif isinstance(self.branch2_backbone, DeiT3):
            depth = len(self.branch2_backbone.layers)
            dims2 = [self.branch2_backbone.embed_dim] * depth
            self.branch2_backbone.downsample_ratios = [self.branch2_backbone.patch_size] * depth

        # 插入交互的层的索引 [2, 6, 10, 13, 16, 19, 22, 25, 28, 31, 34, 38]
        self.branch1_interaction_layer_index = [self.branch1_interaction_indexes[idx][-1] for idx in range(num_interactions)]
        self.branch2_interaction_layer_index = [self.branch2_interaction_indexes[idx][-1] for idx in range(num_interactions)]

        self.interactions = nn.Sequential(*[
            TwoBranchInteractionBlock(
                branch1_dim=dims1[self.branch1_interaction_layer_index[idx]], # 2:128 6:256 34:512 38:1024
                branch2_dim=dims2[self.branch2_interaction_layer_index[idx]],
                # 当前交互模块的特征尺寸, 原始尺寸//下采样的倍数  = 512//4=128   512//8=64  512//16=32  512//32=16
                branch1_feat_size=self.branch1_real_size // self.branch1_backbone.downsample_ratios[self.branch1_interaction_layer_index[idx]],
                branch2_feat_size=self.branch2_real_size // self.branch2_backbone.downsample_ratios[self.branch2_interaction_layer_index[idx]],
                num_heads=deform_num_heads, 
                n_points=n_points,
                drop_path=interaction_drop_path_rate,
                norm_layer=norm_layer, 
                with_cffn=with_cffn,
                cffn_ratio=cffn_ratio, 
                deform_ratio=deform_ratio,
                attn_type=interact_attn_type,
                with_proj=interaction_proj,
            )
            for idx in range(len(self.branch1_interaction_indexes))             
        ])

        self.with_simple_fpn = with_simple_fpn # False
        # assert not is_dino
        # import pdb;pdb.set_trace()
        out_dim = self.branch1_backbone.embed_dim  # 输出的维度，这里可以一样，或以分之二为主
        self.branch1_is_cnn = isinstance(self.branch1_backbone, ConvNeXt)
        # H,W of the final output is decided by branch3

        if with_simple_fpn: # ViT simple FPN
            assert not self.branch1_is_cnn
            
            # fpns are 4x, 2x, 1x, 1/2x
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(out_dim, out_dim, 2, 2),
                nn.GroupNorm(32, out_dim),
                nn.GELU(),
                nn.ConvTranspose2d(out_dim, out_dim, 2, 2)
            )
            self.fpn2 = nn.Sequential(nn.ConvTranspose2d(out_dim, out_dim, 2, 2))
            self.fpn3 = nn.Sequential(nn.Identity())
            self.fpn4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))
        
        elif self.branch1_is_cnn: # CNN regular FPN: no FPNs needed for upsampling
            self.fpn1 = nn.Identity() # 什么都不做，占位的
            self.fpn2 = nn.Identity()
            self.fpn3 = nn.Identity()
            self.fpn4 = nn.Identity()

        else: # ViT regular fpn
            # fpns are 4x, 2x, 1x, 1/2x
            dim_fpn1 = dims1[self.branch1_interaction_layer_index[self.out_interaction_indexes[0]]]
            dim_fpn2 = dims1[self.branch1_interaction_layer_index[self.out_interaction_indexes[1]]]
            
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(dim_fpn1, dim_fpn1, 2, 2),
                nn.GroupNorm(32, dim_fpn1),
                nn.GELU(),
                nn.ConvTranspose2d(dim_fpn1, dim_fpn1, 2, 2)
            )
            self.fpn2 = nn.Sequential(nn.ConvTranspose2d(dim_fpn2, dim_fpn2, 2, 2))
            self.fpn3 = nn.Sequential(nn.Identity())
            self.fpn4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))


        self.fpn1.apply(self._init_weights) 
        self.fpn2.apply(self._init_weights)
        self.fpn3.apply(self._init_weights)
        self.fpn4.apply(self._init_weights)

        dim1 = self.branch1_backbone.embed_dim
        dim2 = self.branch2_backbone.embed_dim

        if not with_simple_fpn:
            assert out_interaction_indexes is not None
            for out_idx in self.out_interaction_indexes: 
                dim1_ = dims1[self.branch1_interaction_layer_index[out_idx]] # [2, 6, 10, 13, 16, 19, 22, 25, 28, 31, 34, 38]
                dim2_ = dims2[self.branch2_interaction_layer_index[out_idx]]
                self.create_intermediate_merge_module(out_idx, dim1_, dim2_)
                # intermediate_merging_{0/1/10}_branch1
                # intermediate_merging_{0/1/10}_branch2
                # intermediate_merging_{0/1/10}_w1
                # intermediate_merging_{0/1/10}_w2
            # dim1_: 128 256 512  
            # dim2_: 128 256 512, 如果是不同的模型的话，dim2_的维度比dim1_的小，将dim2_升维到dim1_
            self.merge_branch1 = nn.Sequential(
                nn.GroupNorm(32, dim1),  #dim1: 1024
                nn.ReLU(inplace=True),
            )                            #dim2: 768 / 1024
            
            if dim2 != dim1:
                self.merge_branch2 = nn.Sequential(
                    nn.GroupNorm(32, dim1),
                    nn.ReLU(inplace=True),
                )
            else:
                self.merge_branch2 = nn.Sequential(
                    nn.GroupNorm(32, dim1),
                    nn.ReLU(inplace=True),
                )

        else: # ViT的话
            # 将branch1的维度变到dim1, ViT
            self.merge_branch1 = nn.Sequential(
                nn.Conv2d(dim1, dim1, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(32, dim1),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim1, dim1, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(32, dim1),
                nn.ReLU(inplace=True),
            )
            # 将branch2的维度变到dim1
            self.merge_branch2 = nn.Sequential(
                nn.Conv2d(dim2, dim1, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(32, dim1),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim1, dim1, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(32, dim1),
                nn.ReLU(inplace=True),
            )


        self.merge_branch1.apply(self._init_weights)
        self.merge_branch2.apply(self._init_weights)

        out_dim = dim1
        self.is_dino = is_dino

        self.fpn2.apply(self._init_weights)
        self.fpn3.apply(self._init_weights)
        self.fpn4.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)

        
        # self.init_weights(pretrained)


    def create_intermediate_merge_module(self, idx, dim1_, dim2_):
        merge_branch1 = nn.Sequential(
            nn.GroupNorm(32, dim1_),  # 128维度，32个通道，GroupNorm一下，提升稳定性。
            nn.ReLU(inplace=True),
        )

        if dim2_ != dim1_:
            merge_branch2 = nn.Sequential(  # 将branch2的维度升到dim1, 如果相同的话，不做这个操作
                nn.Conv2d(dim2_, dim1_, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(32, dim1_),  
                nn.ReLU(inplace=True),
            )
        else:
            merge_branch2 = nn.Sequential(  # 将branch2的维度升到dim1, 如果相同的话，不做这个操作
                nn.GroupNorm(32, dim1_),  
                nn.ReLU(inplace=True),
            )

        merge_branch1.apply(self._init_weights)
        merge_branch2.apply(self._init_weights)

        w1 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        w2 = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        
        setattr(self, f"intermediate_merging_{idx}_branch1", merge_branch1)  #做一下groupNorm
        setattr(self, f"intermediate_merging_{idx}_branch2", merge_branch2)  #做一下groupNorm和升维度

        setattr(self, f"intermediate_merging_{idx}_w1", w1)  # merge两个的权重
        setattr(self, f"intermediate_merging_{idx}_w2", w2)  # merge两个的权重

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    def _get_pos_embed(self, pos_embed, pretrain_size, patch_size, H, W):  # [1, 576, 1024], 24, 24  
        # import pdb; pdb.set_trace()
        # [1,196,1024]-->[1,14,14,1024]-->[1,1024,14,14]--[1,1024,24,24] 理想的，但是vit_models那里,如果patchembedding设置成
        pos_embed = pos_embed.reshape(1, pretrain_size[0] // patch_size[0], pretrain_size[1] // patch_size[1], -1).permute(0, 3, 1, 2)
        
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed.type(self.dtype)

    def _init_deform_weights(self, m):
        if has_deform_attn:
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    @property
    def dtype(self):
        if isinstance(self.branch1_backbone, ConvNeXt):
            return self.branch1_backbone.downsample_layers[0][0].weight.dtype
        else:
            return self.branch1_backbone.patch_embed.proj.weight.dtype

    def forward(self, x):

        # 在此确保，branch1_backbone 的参数不会被更新，仅作推理和交互，并提供特征图
        self.branch1_backbone.eval()
        for param in self.branch1_backbone.parameters():
            param.requires_grad = False
        
        x_origin = x.clone() # [2, 10, 640, 640]
        branch1_output_list = self.branch1_backbone(x_origin) # [1,128,128,128] [1,256,64,64] [1,512,32,32] [1,1024,16,16]

        # 分支1的x：x1
        x1 = x.clone()
        x2 = x.clone()

        deform_inputs_list = []
        for idx in range(len(self.interactions)): # 0 ~ 12个插入模块
            deform_inputs = {}                                             # 交互模块的位置
            if self.interact_attn_type == "deform":                        # [2, 6, 10, 13, 16, 19, 22, 25, 28, 31, 34, 38]
                downsample_ratio1 = self.branch1_backbone.downsample_ratios[self.branch1_interaction_layer_index[idx]]
                downsample_ratio2 = self.branch2_backbone.downsample_ratios[self.branch2_interaction_layer_index[idx]]
                deform_inputs["2to1"] = deform_inputs_1_vit(x1, x2, downsample_ratio1, downsample_ratio2)  # x2的尺寸，x1生成参考点
                deform_inputs["1to2"] = deform_inputs_2_vit(x2, x1, downsample_ratio1, downsample_ratio2)  # x1的尺寸，x2生成参考点
            else:
                deform_inputs["2to1"] = [None, None, None]
                deform_inputs["1to2"] = [None, None, None]
            deform_inputs_list.append(deform_inputs)


        # import pdb;pdb.set_trace()
        # Patch embedding and position embedding
        if isinstance(self.branch1_backbone, ConvNeXt):
            x1 = self.branch1_backbone.downsample_layers[0](x1) # [2,10,512,512] --> [2,128,128,128]# 第一个下采样层
            bs1, _, H1, W1 = x1.shape # here dim is not the final dim1 [2,128,128,128]
            x1 = x1.view(bs1, -1, H1*W1).transpose(1, 2) # (B,C,H,W) -> (B,C,N)-> (B,N,C) # [2,128,128,128]->[2,128,16384]->[1,16384,128]
        elif isinstance(self.branch1_backbone, DeiT3):
            x1, H1, W1 = self.branch1_backbone.patch_embed(x1) # [2, 576, 1024],24,24  (384/16)^2=24^2=576
            bs1, n1, dim1 = x1.shape # 2, 576, 1024
            if self.branch1_backbone.pos_embed is not None:
                pos_embed1 = self.branch1_backbone.pos_embed if not self.branch1_w_cls_token else self.branch1.pos_embed[:, 1:]  # [1, 576, 1024]
                pos_embed1 = self._get_pos_embed(pos_embed1.float(), (self.branch1_backbone.pretrain_img_size, self.branch1_backbone.pretrain_img_size),  
                                                (self.branch1_backbone.patch_size, self.branch1_backbone.patch_size), H1, W1) 
                x1 = x1 + pos_embed1
            x1 = self.branch1_backbone.pos_drop(x1)
        
        if isinstance(self.branch2_backbone, ConvNeXt):
            x2 = self.branch2_backbone.downsample_layers[0](x2)        # [2,10,512,512] --> [2,128,128,128]# 第一个下采样层
            bs2, _, H2, W2 = x2.shape # here dim is not the final dim2 [2,128,128,128]
            x2 = x2.view(bs2, -1, H2*W2).transpose(1, 2) # (B,C,H,W) -> (B,C,N)-> (B,N,C)
        elif isinstance(self.branch2_backbone, DeiT3):
            x2, H2, W2 = self.branch2_backbone.patch_embed(x1) # [2, 576, 1024],24,24  (384/16)^2=24^2=576
            bs2, n2, dim2 = x2.shape # 2, 576, 1024
            if self.branch2_backbone.pos_embed is not None:
                pos_embed2 = self.branch2_backbone.pos_embed if not self.branch1_w_cls_token else self.branch1.pos_embed[:, 1:]  # [1, 576, 1024]
                pos_embed2 = self._get_pos_embed(pos_embed2.float(), (self.branch2_backbone.pretrain_img_size, self.branch2_backbone.pretrain_img_size),  
                                                (self.branch2_backbone.patch_size, self.branch2_backbone.patch_size), H1, W1) 
                x2 = x2 + pos_embed2
            x2 = self.branch2.pos_drop(x2)

        outs = [] # 这里的x1 和 x2 已经是过完模型embedding的了
        # Blocks and interactions
        for i, layer in enumerate(self.interactions): # 12个
            indexes1 = self.branch1_interaction_indexes[i] # [0,2] [3,6] [7,10] [11,13]
            branch1_blocks = self.branch1_backbone.blocks[indexes1[0]:indexes1[-1] + 1]\
                if isinstance(self.branch1_backbone, ConvNeXt) else self.branch1_backbone.layers[indexes1[0]:indexes1[-1] + 1]
            # 可能叫这个：downsample_block convnext的block: convnext_block convnext_block convnext_block 

            indexes2 = self.branch2_interaction_indexes[i]
            branch2_blocks = self.branch2_backbone.blocks[indexes2[0]:indexes2[-1] + 1]\
                if isinstance(self.branch2_backbone, ConvNeXt) else self.branch2_backbone.layers[indexes2[0]:indexes2[-1] + 1]
            

            # 需要反馈回新的H1,W1,H2,W2 因为下采样了,更新长宽
            x1, x2, _, _, H1, W1, H2, W2 = layer(x1, x2,     #[1,16384,128] [1,16384,128]
                        branch1_blocks, branch2_blocks,      # 3个convnext bolck 3个 convnext block
                        H1=H1, W1=W1, H2=H2, W2=W2,          # 128,128  128,128
                        cls1=None, cls2=None,
                        deform_inputs=deform_inputs_list[i])

            # 顺利跑通！

            # import pdb;pdb.set_trace()

            # 代表是CNN的convnext，多个block之后，就要merge一下。
            if not self.with_simple_fpn and i in self.out_interaction_indexes:  
                # import pdb;pdb.set_trace()                        # 0 1 2  3456  7:34  35:38
                # 走这个, 在第几个interaction后输出。[0,1,10]       # 3     3(4)  27(28) 3(4) = 36层+3个下采样层
                # 共插入了12个交互模块， 0,1,10 3个输出。
                # 0: block[0],block[1] block[2] blocks[0:2]              # stage 1
                # 1: block[3],block[4] block[5] block[6]    blocks[3:6]  # stage 2
                # 10:block[32] block[33] block[34] blocks[32:34]         # stage 3 
                x1_ = x1.transpose(1, 2).view(bs1, -1, H1, W1) # [2,16384,128]-->[2,128,128,128]
                x1_ = getattr(self, f"intermediate_merging_{i}_branch1")(x1_) # 单纯的GroupNorm一下 [2, 128, 128, 128]
                x1_ = x1_.type(torch.float32)
                x1_ = F.interpolate(x1_, size=(H2, W2), mode='bilinear', align_corners=False) # 不动
                x1_ = x1_.type(self.dtype)

                x2_ = x2.transpose(1, 2).view(bs2, -1, H2, W2) # [1,16384,128]-->[1,128,128,128]
                x2_ = getattr(self, f"intermediate_merging_{i}_branch2")(x2_) # x2的维度，变道和x1一样，最高,GroupNorm一下
                x2_ = x2_.type(torch.float32)
                # x2_ = F.interpolate(x2_, size=(H2, W2), mode='bilinear', align_corners=False) # 不动
                # x2_ = x2_.type(self.dtype)

                # [1,128,128,128] 两个分支合并起来
                cur_out = x1_ * getattr(self, f"intermediate_merging_{i}_w1") +\
                          x2_ * getattr(self, f"intermediate_merging_{i}_w2")

                outs.append(cur_out) #      

        # import pdb;pdb.set_trace()
        # 最终的分支合并。
        # Branch merging
        # 最终的x1: [2,256,1024]-->[2,1024,16,16]
        x1 = x1.transpose(1, 2).view(bs1, self.branch1_backbone.embed_dim, H1, W1) # [2, 256, 1024] --> [2, 1024, 16, 16]
        x1 = self.merge_branch1(x1)   # 特征图维度变到branch1的 [2, 1024, 16, 16]->[2, 1024, 16, 16]            
        x1 = x1.type(torch.float32) 
        x1 = F.interpolate(x1, size=(H2, W2), mode='bilinear', align_corners=False)  # 特征图尺寸变到branch2的(不动)
        x1 = x1.type(self.dtype) # [2, 1024, 24, 24]->[2, 1024, 40, 40]

        x2 = x2.transpose(1, 2).view(bs2, self.branch2_backbone.embed_dim, H2, W2) # [2, 1024, 768] -> [2, 768, 32, 32]
        x2 = self.merge_branch2(x2)  # 特征图维度变到branch1的 [2, 768, 32, 32]->[2, 1024, 32, 32]
        x2 = x2.type(torch.float32)
        # x2 = F.interpolate(x2, size=(H2, W2), mode='bilinear', align_corners=False) # 特征图尺寸变到branch3的
        # x2 = x2.type(self.dtype) # [2, 1024, 32, 32]->[2, 1024, 40, 40]
      
        out = x1 * self.w1 + x2 * self.w2  # 最终的输出, x1的特征和x2的特征共同作用！ # [2,1024,16,16]
        # 对于convnext，这里应该是一个多尺度的输出。而不是一个普通

        if self.cal_flops:
            return

        if self.with_simple_fpn: # ViT or CNN simple fpn
            f1 = self.fpn1(out).contiguous().float()
            f2 = self.fpn2(out).contiguous().float()
            f3 = self.fpn3(out).contiguous().float()
            f4 = self.fpn4(out).contiguous().float()
        else:
            outs.append(out)
            assert len(outs) == 4
            f1 = self.fpn1(outs[0]).contiguous().float() # self.fpn1: Identity()
            f2 = self.fpn2(outs[1]).contiguous().float()
            f3 = self.fpn3(outs[2]).contiguous().float()
            f4 = self.fpn4(outs[3]).contiguous().float()

        
        branch2_output_list = [f1, f2, f3, f4] # [1,128,128,128] [1,256,64,64] [1,512,32,32] [1,1024,16,16]

        return branch1_output_list, branch2_output_list
        

