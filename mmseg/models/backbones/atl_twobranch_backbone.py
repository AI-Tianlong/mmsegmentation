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

try:
    from mmseg.models.ops.deformable_attention.modules import MSDeformAttn

    has_deform_attn = True
except:
    has_deform_attn = False

from .deit import vit_models
from .beit import BEiT
from .internvit_6b import InternViT6B
from .uniperceiver import UnifiedBertEncoder
from .piip_modules import deform_inputs_1_vit, deform_inputs_2_vit, ThreeBranchInteractionBlock


# mmcv 1.x
# from mmdet.models.builder import BACKBONES
# from mmdet.utils import get_root_logger

@MODELS.register_module()
class TwoBranch_backbone(nn.Module):
    def __init__(self,
                 land_use_branch={},
                 new_task_branch={},
                 ):
        
        super().__init__()
        

        if norm_layer == "none":
            norm_layer = nn.Identity

        self.land_use_branch = MODELS.build(land_use_branch)
        self.new_task_branch = MODELS.build(new_task_branch)
 
        
        dim1 = self.branch1.embed_dim  # branch1的维度, 最大
        dim2 = self.branch2.embed_dim  # branch2的维度, 中等
        dim3 = self.branch3.embed_dim  # branch3的维度, 最小
        assert dim1 >= dim2 >= dim3
        
        # 将branch1的维度变到dim1
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
        # 将branch3的维度变到dim1
        self.merge_branch3 = nn.Sequential(
            nn.Conv2d(dim3, dim1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, dim1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim1, dim1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, dim1),
            nn.ReLU(inplace=True),
        )
        
        self.merge_branch1.apply(self._init_weights)
        self.merge_branch2.apply(self._init_weights)
        self.merge_branch3.apply(self._init_weights)
        
        out_dim = dim1
        self.is_dino = is_dino
        if not is_dino: # 如果不是dino，则输出4个特征图
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(out_dim, out_dim, 2, 2),
                nn.GroupNorm(32, out_dim),
                nn.GELU(),
                nn.ConvTranspose2d(out_dim, out_dim, 2, 2)
            )
            self.fpn1.apply(self._init_weights) 
            
        self.fpn2 = nn.Sequential(nn.ConvTranspose2d(out_dim, out_dim, 2, 2))
        self.fpn3 = nn.Sequential(nn.Identity())
        self.fpn4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))


        self.fpn2.apply(self._init_weights)
        self.fpn3.apply(self._init_weights)
        self.fpn4.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        self.init_weights(pretrained)
        
    @property
    def dtype(self):
        return self.branch3.patch_embed.proj.weight.dtype
    
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
                
    def init_weights(self, pretrained=None):

        if isinstance(pretrained, str):
            # logger = get_root_logger()
            checkpoint = torch.load(pretrained, map_location='cpu')
            if 'model' in checkpoint:
                checkpoint = checkpoint['model']
            if 'module' in checkpoint:
                checkpoint_old = checkpoint['module']
                checkpoint = {}
                for k, v in checkpoint_old.items():
                    checkpoint[k.replace('backbone.', '')] = v
            message = self.load_state_dict(checkpoint, strict=False)
            print_log(message)

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

    def forward(self, x):
        # Resize images
        # 根据branch3的图像尺寸，算1和3的缩放因子
        scale_factor_1to3 = self.branch1_real_size / self.branch3_real_size
        if scale_factor_1to3 < 1:
            x1 = F.interpolate(x, scale_factor=scale_factor_1to3, mode='bilinear', align_corners=False)
        else:
            x1 = x.clone()
        
        # 根据branch3的图像尺寸，算2和3的缩放因子
        scale_factor_2to3 = self.branch2_real_size / self.branch3_real_size
        if scale_factor_2to3 < 1:
            x2 = F.interpolate(x, scale_factor=scale_factor_2to3, mode='bilinear', align_corners=False)
        else:
            x2 = x.clone()

        x3 = x.clone()
        
        # import pdb; pdb.set_trace()

        x1 = x1.type(self.dtype) # [2, 4, 384, 384]
        x2 = x2.type(self.dtype) # [2, 4, 512, 512]
        x3 = x3.type(self.dtype) # [2, 4, 640, 640]

        deform_inputs = {}
        if self.interact_attn_type == "deform":                                              
            deform_inputs["2to1"] = deform_inputs_1_vit(x1, x2)  # 1和2的deform输入  [[1,576 ,1,2]], [32,32], 0]  [reference_points, spatial_shapes, level_start_index]
            deform_inputs["1to2"] = deform_inputs_2_vit(x2, x1)  # 2和1的deform输入  [[1,1024,1,2]], [24,24], 0]  [reference_points, spatial_shapes, level_start_index]
            deform_inputs["3to2"] = deform_inputs_1_vit(x2, x3)  # 2和3的deform输入  [[1,1024,1,2]], [40,40], 0]  [reference_points, spatial_shapes, level_start_index]
            deform_inputs["2to3"] = deform_inputs_2_vit(x3, x2)  # 3和2的deform输入  [[1,1600,1,2]], [32,32], 0]  [reference_points, spatial_shapes, level_start_index]
        else:
            deform_inputs["2to1"] = [None, None, None]
            deform_inputs["1to2"] = [None, None, None]
            deform_inputs["3to2"] = [None, None, None]
            deform_inputs["2to3"] = [None, None, None]
        
        # import pdb; pdb.set_trace()
        # Patch embedding and position embedding
        if 'perceiver' in self.branch1.pretrained:
            x1, H1, W1 = self.branch1.visual_embed(x1)
            bs1, n1, dim1 = x1.shape
        else:
            x1, H1, W1 = self.branch1.patch_embed(x1) # [2, 576, 1024],24,24  (384/16)^2=24^2=576
            bs1, n1, dim1 = x1.shape # 2, 576, 1024
            if self.branch1.pos_embed is not None:
                pos_embed1 = self.branch1.pos_embed if not self.branch1_w_cls_token else self.branch1.pos_embed[:, 1:]  # [1, 576, 1024]
                pos_embed1 = self._get_pos_embed(pos_embed1.float(), (self.branch1.pretrain_img_size, self.branch1.pretrain_img_size),  
                                                (self.branch1.patch_size, self.branch1.patch_size), H1, W1) 
                x1 = x1 + pos_embed1
            x1 = self.branch1.pos_drop(x1)

        if 'perceiver' in self.branch2.pretrained:
            x2, H2, W2 = self.branch2.visual_embed(x2)
            bs2, n2, dim2 = x2.shape
        else:
            x2, H2, W2 = self.branch2.patch_embed(x2) # [2, 1024, 768], 32,32 512/16=32 
            bs2, n2, dim2 = x2.shape
            if self.branch2.pos_embed is not None:
                pos_embed2 = self.branch2.pos_embed if not self.branch2_w_cls_token else self.branch2.pos_embed[:, 1:]
                pos_embed2 = self._get_pos_embed(pos_embed2.float(), (self.branch2.pretrain_img_size, self.branch2.pretrain_img_size), 
                                                (self.branch2.patch_size, self.branch2.patch_size), H2, W2) 
                x2 = x2 + pos_embed2
            x2 = self.branch2.pos_drop(x2)

        if 'perceiver' in self.branch3.pretrained:
            x3, H3, W3 = self.branch3.visual_embed(x3)
            bs3, n3, dim3 = x3.shape
        else:
            x3, H3, W3 = self.branch3.patch_embed(x3) # [2,1600,384],40,40 640/16=40
            bs3, n3, dim3 = x3.shape
            if self.branch3.pos_embed is not None:
                pos_embed3 = self.branch3.pos_embed if not self.branch3_w_cls_token else self.branch3.pos_embed[:, 1:]
                pos_embed3 = self._get_pos_embed(pos_embed3.float(), (self.branch3.pretrain_img_size, self.branch3.pretrain_img_size), 
                                                (self.branch3.patch_size, self.branch3.patch_size), H3, W3) 
                x3 = x3 + pos_embed3
            x3 = self.branch3.pos_drop(x3)


        # Blocks and interactions
        for i, layer in enumerate(self.interactions):
            indexes1 = self.branch1_interaction_indexes[i]
            branch1_blocks = self.branch1.blocks[indexes1[0]:indexes1[-1] + 1]\
                if 'perceiver' not in self.branch1.pretrained else self.branch1.layers[indexes1[0]:indexes1[-1] + 1]
            indexes2 = self.branch2_interaction_indexes[i]
            branch2_blocks = self.branch2.blocks[indexes2[0]:indexes2[-1] + 1]\
                if 'perceiver' not in self.branch2.pretrained else self.branch2.layers[indexes2[0]:indexes2[-1] + 1]
            indexes3 = self.branch3_interaction_indexes[i]
            branch3_blocks = self.branch3.blocks[indexes3[0]:indexes3[-1] + 1]\
                if 'perceiver' not in self.branch3.pretrained else self.branch3.layers[indexes3[0]:indexes3[-1] + 1]

            x1, x2, x3, _, _, _ = layer(x1, x2, x3,
                        branch1_blocks, branch2_blocks, branch3_blocks,
                        H1=H1, W1=W1, H2=H2, W2=W2, H3=H3, W3=W3,
                        cls1=None, cls2=None, cls3=None,
                        deform_inputs=deform_inputs)

        # merge前的特征尺寸
        # branch1: [2,576,1024]  # 在计算过程中 是一直不变的 
        # branch2: [2,1024,768]  # 在计算过程中 是一直不变的
        # branch3: [2,1600,384]  # 在计算过程中 是一直不变的

        # Branch merging
        x1 = x1.transpose(1, 2).view(bs1, dim1, H1, W1) # [2, 576, 1024] --> [2, 1024, 24, 24]
        x1 = self.merge_branch1(x1)   # 特征图维度变到branch1的 [2, 1024, 24, 24]->[2, 1024, 24, 24]            
        x1 = x1.type(torch.float32) 
        x1 = F.interpolate(x1, size=(H3, W3), mode='bilinear', align_corners=False)  # 特征图尺寸变到branch3的
        x1 = x1.type(self.dtype) # [2, 1024, 24, 24]->[2, 1024, 40, 40]

        x2 = x2.transpose(1, 2).view(bs2, dim2, H2, W2) # [2, 1024, 768] -> [2, 768, 32, 32]
        x2 = self.merge_branch2(x2)  # 特征图维度变到branch1的 [2, 768, 32, 32]->[2, 1024, 32, 32]
        x2 = x2.type(torch.float32)
        x2 = F.interpolate(x2, size=(H3, W3), mode='bilinear', align_corners=False) # 特征图尺寸变到branch3的
        x2 = x2.type(self.dtype) # [2, 1024, 32, 32]->[2, 1024, 40, 40]
        
        x3 = x3.transpose(1, 2).view(bs3, dim3, H3, W3) # [2,1600,384] -> [2, 384, 40, 40]
        x3 = self.merge_branch3(x3)  # 特征图维度变到branch1的 
        
        out = x1 * self.w1 + x2 * self.w2 + x3 * self.w3  # 最终的输出
             
        
        # Outputs for fpn
        if not self.is_dino:
            f1 = self.fpn1(out).contiguous().float() # [2, 1024, 160, 160]
            f2 = self.fpn2(out).contiguous().float() # [2, 1024, 80, 80]
            f3 = self.fpn3(out).contiguous().float() # [2, 1024, 40, 40]
            f4 = self.fpn4(out).contiguous().float() # [2, 1024, 20, 20]
            return [f1, f2, f3, f4]
            
        else:
            f2 = self.fpn2(out).contiguous().float()
            f3 = self.fpn3(out).contiguous().float()
            f4 = self.fpn4(out).contiguous().float()
            return [f2, f3, f4]