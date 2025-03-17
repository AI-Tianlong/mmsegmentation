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
class PIIPThreeBranch_MultiModal(nn.Module):
    def __init__(self,
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
        
        self.interact_attn_type = interact_attn_type
        
        # 三个通道前的权重
        self.w1 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.w2 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.w3 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        
        self.branch1_interaction_indexes = branch1.pop("interaction_indexes")
        self.branch2_interaction_indexes = branch2.pop("interaction_indexes")
        self.branch3_interaction_indexes = branch3.pop("interaction_indexes")
        
        self.branch1_real_size = branch1.pop("real_size")
        self.branch2_real_size = branch2.pop("real_size")
        self.branch3_real_size = branch3.pop("real_size")
        
        self.branch1_w_cls_token = branch1.pop("branch1_w_cls_token", False)
        self.branch2_w_cls_token = branch2.pop("branch2_w_cls_token", False)
        self.branch3_w_cls_token = branch3.pop("branch3_w_cls_token", False)
        
        if 'deit' in branch1['pretrained']:
            self.branch1 = vit_models(**branch1)
        elif 'beit' in branch1['pretrained']: # 如果beit这个词在pretrain中，则构建BEiT模型
            self.branch1 = BEiT(**branch1)
        elif 'perceiver' in branch1['pretrained']:
            self.branch1 = UnifiedBertEncoder(**branch1)
        else:
            self.branch1 = InternViT6B(**branch1)
            self.branch1_w_cls_token = True
        
        if 'deit' in branch2['pretrained']:
            self.branch2 = vit_models(**branch2)
        elif 'beit' in branch2['pretrained']:
            self.branch2 = BEiT(**branch2)
        elif 'perceiver' in branch2['pretrained']:
            self.branch2 = UnifiedBertEncoder(**branch2)
        else:
            self.branch2 = InternViT6B(**branch2)
            self.branch2_w_cls_token = True
            
        if 'deit' in branch3['pretrained']:
            self.branch3 = vit_models(**branch3)
        elif 'beit' in branch3['pretrained']:
            self.branch3 = BEiT(**branch3)
        elif 'perceiver' in branch3['pretrained']:
            self.branch3 = UnifiedBertEncoder(**branch3)
        else:
            self.branch3 = InternViT6B(**branch3)
            self.branch3_w_cls_token = True
        
        assert len(self.branch1_interaction_indexes) == len(self.branch2_interaction_indexes) == len(self.branch3_interaction_indexes)
        self.interactions = nn.Sequential(*[
            ThreeBranchInteractionBlock(
                branch1_dim=self.branch1.embed_dim, #1024
                branch2_dim=self.branch2.embed_dim, #768
                branch3_dim=self.branch3.embed_dim, #384

                branch1_img_size=self.branch1.pretrain_img_size,  # 224 ？
                branch2_img_size=self.branch2.pretrain_img_size,  # 224 ？
                branch3_img_size=self.branch3.pretrain_img_size,  # 224 ？

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
            for _ in range(len(self.branch1_interaction_indexes))
        ])
        
        dim1 = self.branch1.embed_dim  # branch1的维度, 最大 1024
        dim2 = self.branch2.embed_dim  # branch2的维度, 中等 768
        dim3 = self.branch3.embed_dim  # branch3的维度, 最小 384
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

        # 原版的PIIP, 这里是同一个图像，然后进行下采样，得到三个输入分支。
        # 但是我这里，我的输入是分别来自于三个不同的图像，所以这里的x是一个list，分别是三个图像的输入。
        # 这里是通过 Encoder-Decoder 来传的图像。

        # Resize images
        # 根据branch3的图像尺寸，算1和3的缩放因子
        # 我这的话，我三个输入，分别是三个不同的图像，如果单模态的话，没有交互，如果多模态的话，则有交互。
        
        # 最后的特征维度，和branch1相同 ---> 特征的维度最大
        # 最后的特征大小，和branch2相同 ---> 不太大 不太小
                                                        # 要不在降采样到1024啊？ 也合理
        # 传的是个张量列表，[2,10,2048,2048], [2,4,640,640], [2,10,224,224]
        # import pdb; pdb.set_trace()
        x1, x2, x3 = x[2], x[1], x[0] # [2,10,224,224],[2, 4, 640, 640],[2, 3, 2048, 2048]]


        # x1 = F.interpolate(x1, size=(224,  224),  mode='bilinear',   align_corners=False)
        # x2 = F.interpolate(x1, size=(640,  640),  mode='bilinear',   align_corners=False)
        # x3 = F.interpolate(x1, size=(2048, 2048), mode='bilinear',   align_corners=False)


        x1 = x1.type(self.dtype)
        x2 = x2.type(self.dtype)
        x3 = x3.type(self.dtype)

        deform_inputs = {}
        if self.interact_attn_type == "deform":
            deform_inputs["2to1"] = deform_inputs_1_vit(x1, x2)  # 1和2的deform输入 # [1, 196, 1, 2]
            deform_inputs["1to2"] = deform_inputs_2_vit(x2, x1)  # 2和1的deform输入
            deform_inputs["3to2"] = deform_inputs_1_vit(x2, x3)  # 2和3的deform输入
            deform_inputs["2to3"] = deform_inputs_2_vit(x3, x2)  # 3和2的deform输入
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
            x1, H1, W1 = self.branch1.patch_embed(x1) # [2, 196, 1024],14,14
            bs1, n1, dim1 = x1.shape # 2, 196, 1024
            if self.branch1.pos_embed is not None:
                pos_embed1 = self.branch1.pos_embed if not self.branch1_w_cls_token else self.branch1.pos_embed[:, 1:] # 如果有cls_token 则195维度
                pos_embed1 = self._get_pos_embed(pos_embed1.float(), (self.branch1.pretrain_img_size, self.branch1.pretrain_img_size),  
                                                (self.branch1.patch_size, self.branch1.patch_size), H1, W1) 
                x1 = x1 + pos_embed1
            x1 = self.branch1.pos_drop(x1)

        if 'perceiver' in self.branch2.pretrained:
            x2, H2, W2 = self.branch2.visual_embed(x2)
            bs2, n2, dim2 = x2.shapes
        else:
            x2, H2, W2 = self.branch2.patch_embed(x2) # [2, 1600, 768] 40 40
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
            x3, H3, W3 = self.branch3.patch_embed(x3) # [2, 16384, 384], 128, 128
            bs3, n3, dim3 = x3.shape
            if self.branch3.pos_embed is not None:
                pos_embed3 = self.branch3.pos_embed if not self.branch3_w_cls_token else self.branch3.pos_embed[:, 1:]
                pos_embed3 = self._get_pos_embed(pos_embed3.float(), (self.branch3.pretrain_img_size, self.branch3.pretrain_img_size), 
                                                (self.branch3.patch_size, self.branch3.patch_size), H3, W3) 
                x3 = x3 + pos_embed3
            x3 = self.branch3.pos_drop(x3)


        # Blocks and interactions
        for i, layer in enumerate(self.interactions):
            indexes1 = self.branch1_interaction_indexes[i] # [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21], [22, 23]]
            branch1_blocks = self.branch1.blocks[indexes1[0]:indexes1[-1] + 1]\
                if 'perceiver' not in self.branch1.pretrained else self.branch1.layers[indexes1[0]:indexes1[-1] + 1]
            indexes2 = self.branch2_interaction_indexes[i] # [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10], [11, 11]]
            branch2_blocks = self.branch2.blocks[indexes2[0]:indexes2[-1] + 1]\
                if 'perceiver' not in self.branch2.pretrained else self.branch2.layers[indexes2[0]:indexes2[-1] + 1]
            indexes3 = self.branch3_interaction_indexes[i] # [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10], [11, 11]]
            branch3_blocks = self.branch3.blocks[indexes3[0]:indexes3[-1] + 1]\
                if 'perceiver' not in self.branch3.pretrained else self.branch3.layers[indexes3[0]:indexes3[-1] + 1]

            x1, x2, x3, _, _, _ = layer(x1, x2, x3,
                                        branch1_blocks, branch2_blocks, branch3_blocks,
                                        H1=H1, W1=W1, H2=H2, W2=W2, H3=H3, W3=W3,
                                        cls1=None, cls2=None, cls3=None,
                                        deform_inputs=deform_inputs)

        # Branch merging
        x1 = x1.transpose(1, 2).view(bs1, dim1, H1, W1) # [2, 196, 1024] --> [2, 1024, 14, 14]
        x1 = self.merge_branch1(x1)   # 特征图维度变到branch1的 [2, 1024, 14, 14]->[2, 1024, 14, 14]            
        x1 = x1.type(torch.float32) 
        x1 = F.interpolate(x1, size=(H2, W2), mode='bilinear', align_corners=False)  # 特征图尺寸变到branch2的 # [2, 1024, 14, 14] -> [2, 1024, 40, 40]
        x1 = x1.type(self.dtype) 

        x2 = x2.transpose(1, 2).view(bs2, dim2, H2, W2) # [2, 1600, 768] -> [2, 768, 40, 40]
        x2 = self.merge_branch2(x2)  # 特征图维度变到branch1的 [2, 768, 40, 40]->[2, 1024, 40, 40]
        x2 = x2.type(torch.float32)
        x2 = F.interpolate(x2, size=(H2, W2), mode='bilinear', align_corners=False) # 特征图尺寸变到branch2的
        x2 = x2.type(self.dtype) # [2, 1024, 40, 40]->[2, 1024, 40, 40]
        
        x3 = x3.transpose(1, 2).view(bs3, dim3, H3, W3) # [2, 16384, 384] -> [2, 384, 128, 128]
        x3 = self.merge_branch3(x3)  # 特征图维度变到branch1的  [2, 384, 128, 128]-->[2,1024,128,128]
        x3 = x3.type(torch.float32)
        x3 = F.interpolate(x3, size=(H2, W2), mode='bilinear', align_corners=False) # 特征图尺寸变到branch2的
        x3 = x3.type(self.dtype) # [2,1024,128,128]->[2, 1024, 40, 40]

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