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


from mmengine.runner.checkpoint import load_state_dict
from .deit import vit_models
from .beit import BEiT
from .internvit_6b import InternViT6B
from .uniperceiver import UnifiedBertEncoder
from .piip_modules import deform_inputs_1_vit, deform_inputs_2_vit, ThreeBranchInteractionBlock

# mmcv 1.x
# from mmdet.models.builder import BACKBONES
# from mmdet.utils import get_root_logger

@MODELS.register_module()
class PIIPThreeBranch_SingleModal(nn.Module):
    def __init__(self,
                 single_branch = 'branch2',
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
                 
                 branch1={},
                 branch2={},
                 branch3={},
                 pretrained=None
                 ):
        
        super().__init__()
        
        if norm_layer == "none":
            norm_layer = nn.Identity
        self.single_branch = single_branch
        self.interact_attn_type = interact_attn_type
        
        # 三个通道前的权重
        self.w1 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.w2 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.w3 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        
        # self.branch1_interaction_indexes = branch1.pop("interaction_indexes")
        # self.branch2_interaction_indexes = branch2.pop("interaction_indexes")
        # self.branch3_interaction_indexes = branch3.pop("interaction_indexes")
        # import pdb; pdb.set_trace()

        if self.single_branch == 'branch1':
            self.branch1_real_size = branch1.pop("real_size")
            self.branch1_w_cls_token = branch1.pop("branch1_w_cls_token", False)
            if 'deit' in branch1['pretrained']:
                self.branch1 = vit_models(**branch1)
            elif 'beit' in branch1['pretrained']: # 如果beit这个词在pretrain中，则构建BEiT模型
                self.branch1 = BEiT(**branch1)
            elif 'perceiver' in branch1['pretrained']:
                self.branch1 = UnifiedBertEncoder(**branch1)
            else:
                self.branch1 = InternViT6B(**branch1)
                self.branch1_w_cls_token = True

        elif self.single_branch == 'branch2':
            self.branch2_real_size = branch2.pop("real_size")
            self.branch2_w_cls_token = branch2.pop("branch2_w_cls_token", False)
            if 'deit' in branch2['pretrained']:
                self.branch2 = vit_models(**branch2)
            elif 'beit' in branch2['pretrained']:
                self.branch2 = BEiT(**branch2)
            elif 'perceiver' in branch2['pretrained']:
                self.branch2 = UnifiedBertEncoder(**branch2)
            else:
                self.branch2 = InternViT6B(**branch2)
                self.branch2_w_cls_token = True

        elif self.single_branch == 'branch3':
            self.branch3_real_size = branch3.pop("real_size")
            self.branch3_w_cls_token = branch3.pop("branch3_w_cls_token", False)
            if 'deit' in branch3['pretrained']:
                self.branch3 = vit_models(**branch3)
            elif 'beit' in branch3['pretrained']:
                self.branch3 = BEiT(**branch3)
            elif 'perceiver' in branch3['pretrained']:
                self.branch3 = UnifiedBertEncoder(**branch3)
            else:
                self.branch3 = InternViT6B(**branch3)
                self.branch3_w_cls_token = True

        else:
            raise TypeError('single_branch must be branch1, branch2 or branch3')

        # import pdb; pdb.set_trace()
        dim1 = 1024  # branch1的维度, 最大 1024
        dim2 = 768  # branch2的维度, 中等 768
        dim3 = 384  # branch3的维度, 最小 384
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
        # self.interactions.apply(self._init_weights)
        # self.apply(self._init_deform_weights)
        self.init_weights(pretrained)
        
    @property
    def dtype(self):
        branch = getattr(self, self.single_branch)
        return branch.patch_embed.proj.weight.dtype
    
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
            message = load_state_dict(self, checkpoint, strict=False, logger='current')
            print_log(message)

    def _get_pos_embed(self, pos_embed, pretrain_size, patch_size, H, W):
        pos_embed = pos_embed.reshape(
            1, pretrain_size[0] // patch_size[0], pretrain_size[1] // patch_size[1], -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed.type(self.dtype)

    def _init_deform_weights(self, m):
        if has_deform_attn:
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    def forward(self, x):

        # import pdb; pdb.set_trace()
        # x1 = F.interpolate(x1, size=(224,  224),  mode='bilinear',   align_corners=False)
        # x2 = F.interpolate(x1, size=(640,  640),  mode='bilinear',   align_corners=False)
        # x3 = F.interpolate(x1, size=(2048, 2048), mode='bilinear',   align_corners=False)

        if self.single_branch == 'branch1':
            branch = self.branch1
        elif self.single_branch == 'branch2':
            branch = self.branch2
        elif self.single_branch == 'branch3':
            branch = self.branch3

        # x = x.type(self.dtype)
        H1=W1 = 224 // 16
        H2=W2 = 640 // 16
        H3=W3 = 1216 // 16
        # import pdb; pdb.set_trace()

        single_out_puts = branch.forward(x) # [1,4,640,640] --> [1,768,40,40]
        single_out_puts = single_out_puts[0]
        bs, dim, H, W = single_out_puts.shape

        
        # Branch merging
        if self.single_branch=='branch1':
            single_out_puts = self.merge_branch1(single_out_puts)   # 特征图维度变到branch1的 [2, 1024, 14, 14]->[2, 1024, 14, 14]            
            single_out_puts = single_out_puts.type(torch.float32) 
            single_out_puts = F.interpolate(single_out_puts, size=(H2, W2), mode='bilinear', align_corners=False)  # 特征图尺寸变到branch2的 # [2, 1024, 14, 14] -> [2, 1024, 40, 40]
            single_out_puts = single_out_puts.type(self.dtype) 
        elif self.single_branch=='branch2':
            single_out_puts = self.merge_branch2(single_out_puts)  # 特征图维度变到branch1的 [2, 768, 40, 40]->[2, 1024, 40, 40]
            single_out_puts = single_out_puts.type(torch.float32)
            single_out_puts = F.interpolate(single_out_puts, size=(H2, W2), mode='bilinear', align_corners=False) # 特征图尺寸变到branch2的
            single_out_puts = single_out_puts.type(self.dtype) # [2, 1024, 40, 40]->[2, 1024, 40, 40]
        elif self.single_branch=='branch3':
            single_out_puts = self.merge_branch3(single_out_puts)  # 特征图维度变到branch1的  [2, 384, 128, 128]-->[2,1024,128,128]
            single_out_puts = single_out_puts.type(torch.float32)
            single_out_puts = F.interpolate(single_out_puts, size=(H2, W2), mode='bilinear', align_corners=False) # 特征图尺寸变到branch2的
            single_out_puts = single_out_puts.type(self.dtype) # [2,1024,128,128]->[2, 1024, 40, 40]
        
        # 这里多模态的特征直接相加？有一些操作是不是会更好的融合？
        out = single_out_puts
        # out = single_out_puts * self.w1 + single_out_puts * self.w2 + single_out_puts * self.w3  # 最终的输出
        
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
    