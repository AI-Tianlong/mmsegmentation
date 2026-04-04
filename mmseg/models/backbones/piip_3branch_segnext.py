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
from .mscan import MSCAN  # segnext

from .piip_modules import (deform_inputs_1_vit, 
                           deform_inputs_2_vit, 
                           ThreeBranchInteractionBlock_segnext)


from .piip_modules import (deform_inputs_1_cnn,
                           deform_inputs_2_cnn, 
                           ThreeBranchInteractionBlock_segnext)
                            
# mmcv 1.x
# from mmdet.models.builder import BACKBONES
# from mmdet.utils import get_root_logger

@MODELS.register_module()
class PIIPThreeBranch_segnext(nn.Module):
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
        
        # 需要把这些值给那啥掉，不然的话，会传递到backbone的初始化去
        self.branch1_interaction_indexes = branch1.pop("interaction_indexes")
        self.branch2_interaction_indexes = branch2.pop("interaction_indexes")
        self.branch3_interaction_indexes = branch3.pop("interaction_indexes")
        
        self.branch1_real_size = branch1.pop("real_size")
        self.branch2_real_size = branch2.pop("real_size")
        self.branch3_real_size = branch3.pop("real_size")
        
        self.branch1_w_cls_token = branch1.pop("branch1_w_cls_token", False)
        self.branch2_w_cls_token = branch2.pop("branch2_w_cls_token", False)
        self.branch3_w_cls_token = branch3.pop("branch3_w_cls_token", False)
        
        self.branch1_pretrain_size = branch1.pop('pretrain_img_size', False)
        self.branch2_pretrain_size = branch2.pop('pretrain_img_size', False)
        self.branch3_pretrain_size = branch3.pop('pretrain_img_size', False)

      
        if 'segnext' in branch1['pretrained']:
            self.branch1 = MSCAN(**branch1)
        else:
            self.branch1 = InternViT6B(**branch1)
            self.branch1_w_cls_token = True
        
     
        if 'segnext' in branch2['pretrained']:
            self.branch2 = MSCAN(**branch2)
        else:
            self.branch2 = InternViT6B(**branch2)
            self.branch2_w_cls_token = True
            
        if 'segnext' in branch3['pretrained']:
            self.branch3 = MSCAN(**branch3)
        else:
            self.branch3 = InternViT6B(**branch3)
            self.branch3_w_cls_token = True
        
        assert len(self.branch1_interaction_indexes) == len(self.branch2_interaction_indexes) == len(self.branch3_interaction_indexes)
        
        self.interactions = nn.Sequential(*[
            ThreeBranchInteractionBlock_segnext(
                branch1_dim=branch1.embed_dims[num_stage],  # 这个还不太通用啊,那就分开写三个呗？
                branch2_dim=branch2.embed_dims[num_stage],
                branch3_dim=branch3.embed_dims[num_stage], 

                branch1_img_size=self.branch1_real_size,
                branch2_img_size=self.branch2_real_size,
                branch3_img_size=self.branch3_real_size,

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
            for num_stage in range(len(self.branch1_interaction_indexes))
        ])
        
        dim1 = branch1.embed_dims  # branch1的维度, 最大
        dim2 = branch2.embed_dims  # branch2的维度, 中等
        dim3 = branch3.embed_dims  # branch3的维度, 最小

        # dim1 是一个列表, 理论上，每一个[64,128,320,512]的元素，都要满足dim1>=dim2>=dim3

        if isinstance(dim1, list):
            for num_stage in range(len(dim1)):
                assert dim1[num_stage] >= dim2[num_stage] >= dim3[num_stage], f'Error: dim1={dim1}, dim2={dim2}, dim3={dim3}'
        elif isinstance(dim1, int):
            assert dim1 >= dim2 >= dim3, f'Error: dim1={dim1}, dim2={dim2}, dim3={dim3}'
        
        # 将branch1的维度变到dim1 # [2,768,24,24]-->[2,1024,24,24] # 这是对于ViT的
        # 维度都一样，都是512 所以这里其实不用给
        # self.merge_branch1 = nn.Sequential(
        #     nn.Conv2d(dim1, dim1, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.GroupNorm(32, dim1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(dim1, dim1, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.GroupNorm(32, dim1),
        #     nn.ReLU(inplace=True),
        # )
        # # 将branch2的维度变到dim1
        # self.merge_branch2 = nn.Sequential(
        #     nn.Conv2d(dim2, dim1, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.GroupNorm(32, dim1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(dim1, dim1, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.GroupNorm(32, dim1),
        #     nn.ReLU(inplace=True),
        # )
        # # 将branch3的维度变到dim1
        # self.merge_branch3 = nn.Sequential(
        #     nn.Conv2d(dim3, dim1, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.GroupNorm(32, dim1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(dim1, dim1, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.GroupNorm(32, dim1),
        #     nn.ReLU(inplace=True),
        # )
        
        # self.merge_branch1.apply(self._init_weights)
        # self.merge_branch2.apply(self._init_weights)
        # self.merge_branch3.apply(self._init_weights)
        
        out_dim = dim1[3]
        self.is_dino = is_dino

        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        self.init_weights(pretrained)
        
    @property
    def dtype(self):
        # import pdb; pdb.set_trace()
        return self.branch1.patch_embed1.proj[0].weight.dtype  #有没有这个参数还是一个问题呢
    
    
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
        outs = []    # 存放最终的金字塔 特征图
        
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
        x1 = x1.type(self.dtype)
        x2 = x2.type(self.dtype)
        x3 = x3.type(self.dtype)

        deform_inputs = {}

        # 不能按照这个来，因为这个是给vit的，用的是原始图像，
        # 这里应该给特征图作为输入

        # Blocks and interactions

        # import pdb; pdb.set_trace()
        for i, layer in enumerate(self.interactions):
            indexes1 = self.branch1_interaction_indexes[i]
            branch1_patch_embed = getattr(self.branch1, f'patch_embed{indexes1[0] + 1}')
            branch1_blocks =  getattr(self.branch1, f'block{indexes1[0]+1}')
            branch1_norm = getattr(self.branch1, f'norm{indexes1[0]+1}')

            indexes2 = self.branch2_interaction_indexes[i]
            branch2_patch_embed = getattr(self.branch2, f'patch_embed{indexes2[0] + 1}')
            branch2_blocks =  getattr(self.branch2, f'block{indexes2[0]+1}')
            branch2_norm = getattr(self.branch2, f'norm{indexes2[0]+1}')

            indexes3 = self.branch3_interaction_indexes[i]
            branch3_patch_embed = getattr(self.branch3, f'patch_embed{indexes3[0] + 1}')
            branch3_blocks =  getattr(self.branch3, f'block{indexes3[0]+1}')
            branch3_norm = getattr(self.branch3, f'norm{indexes3[0]+1}')

            # import pdb; pdb.set_trace()

            # [2, 25600, 64], 160, 160   
            # [2, 6400, 128], 80, 80   
            # [2, 1600, 320], 40, 40   
            # [2, 400, 512], 20, 20   
            B = x.shape[0]
            def _segnext_block_forward(x, patch_embed, block, norm):
                x, H, W = patch_embed(x)
                for blk in block:         # 过 depth 个 block
                    x = blk(x, H, W)     # 不变
                x = norm(x)       
                # x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                return x, H, W

            x1, H1, W1 = _segnext_block_forward(x1, branch1_patch_embed, branch1_blocks,branch1_norm)  # [2,64,56,56],56,56
            x2, H2, W2 = _segnext_block_forward(x2, branch2_patch_embed, branch2_blocks,branch2_norm)  # [2,64,96,96],96,96
            x3, H3, W3 = _segnext_block_forward(x3, branch3_patch_embed, branch3_blocks,branch3_norm)  # [2,64,160,160],160,160
             
            # import pdb; pdb.set_trace()
            if self.interact_attn_type == "deform":
            # 这里，怎么计算 可形变注意力
            # import pdb; pdb.set_trace()
                deform_inputs["2to1"] = deform_inputs_1_cnn(x1, H1, W1, x2, H2, W2)  # 1和2的deform输入
                deform_inputs["1to2"] = deform_inputs_2_cnn(x2, H2, W2, x1, H1, W1)  # 2和1的deform输入
                deform_inputs["3to2"] = deform_inputs_1_cnn(x2, H2, W2, x3, H3, W3)  # 2和3的deform输入
                deform_inputs["2to3"] = deform_inputs_2_cnn(x3, H3, W3, x2, H2, W2)  # 3和2的deform输入
            else:
                deform_inputs["2to1"] = [None, None, None]
                deform_inputs["1to2"] = [None, None, None]
                deform_inputs["3to2"] = [None, None, None]
                deform_inputs["2to3"] = [None, None, None]
        
            # import pdb; pdb.set_trace()
            x1, x2, x3, = layer(x1, x2, x3,
                                H1=H1, W1=W1, H2=H2, W2=W2, H3=H3, W3=W3,
                                deform_inputs=deform_inputs)   # 传给了ThreeBranchInteractionBlock_segnext

            x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()  # [2, 64, 56, 56]
            x2 = x2.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()  # [2, 64, 128, 128]
            x3 = x3.reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous()  # [2, 64, 160, 160]

            # Branch merging
            
            # 通道一样，只变尺度
            x1 = x1.type(torch.float32) 
            x1 = F.interpolate(x1, size=(H3, W3), mode='bilinear', align_corners=False)  # 特征图尺寸变到branch3的
            x1 = x1.type(self.dtype) # [2, 1024, 24, 24]->[2, 1024, 40, 40]

            # 通道一样，只变尺度
            x2 = x2.type(torch.float32)
            x2 = F.interpolate(x2, size=(H3, W3), mode='bilinear', align_corners=False) # 特征图尺寸变到branch3的
            x2 = x2.type(self.dtype) # [2, 1024, 32, 32]->[2, 1024, 40, 40]
            
            x3 = x3.type(torch.float32) 
            x3 = x3.type(self.dtype)
            
            out = x1 * self.w1 + x2 * self.w2 + x3 * self.w3  # 最终的输出
            outs.append(out.contiguous().float()) 
            # import pdb; pdb.set_trace()

        return outs