# --------------------------------------------------------
# PIIP
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from functools import partial
import warnings

import torch
import torch.nn as nn

try:
    from mmseg.models.ops.deformable_attention.modules import MSDeformAttn
    has_deform_attn = True
except:
    has_deform_attn = False
    
from timm.models.layers import DropPath
import torch.utils.checkpoint as cp
from .convnext_hf_wrapper import ConvNextLayerWrapper
from .convnext import ConvNeXtBlock

class Permute(nn.Module):
    def __init__(self, *dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)



def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points

def deform_inputs_1_vit(x1, x2, patch_size1=16, patch_size2=16): # patch_size缩小四倍？
    # query = small image
    # key = large image
    # x1 is small image
    # import pdb;pdb.set_trace()
    _, _, h1, w1 = x1.shape # [1,10,512,512]
    _, _, h2, w2 = x2.shape # [1,10,512,512] 
    spatial_shapes = torch.as_tensor([(h2 // patch_size2, w2 // patch_size2)], dtype=torch.long, device=x1.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h1 // patch_size1, w1 // patch_size1)], x1.device)
    deform_inputs1 = [reference_points, spatial_shapes, level_start_index]
                     # [1,28224,1,2],(224,224),0]
    return deform_inputs1

def deform_inputs_2_vit(x1, x2, patch_size1=16, patch_size2=16):
    # query = large image
    # key = small image
    # x1 is large image
    _, _, h1, w1 = x1.shape
    _, _, h2, w2 = x2.shape
    spatial_shapes = torch.as_tensor([(h2 // patch_size1, w2 // patch_size1)], dtype=torch.long, device=x1.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h1 // patch_size2, w1 // patch_size2)], x1.device)
    deform_inputs2 = [reference_points, spatial_shapes, level_start_index]
    
    return deform_inputs2



class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x).flatten(2).transpose(1, 2)
        return x

class CrossAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            attn_drop=0.,
            proj_drop=0.,
            dim_feat=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        if dim_feat is None:
            dim_feat = dim

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim_feat, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim_feat, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_q, x_kv):
        B, N_q, C = x_q.shape
        _, N_kv, _ = x_kv.shape
        q = self.wq(x_q).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.wk(x_kv).reshape(B, N_kv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.wv(x_kv).reshape(B, N_kv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class Injector(nn.Module):
    def __init__(self, 
                 dim, 
                 num_heads=6, 
                 n_points=4, 
                 n_levels=1, 
                 deform_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                 with_cp=False, with_cffn=False, cffn_ratio=0.25, drop=0., drop_path=0., attn_type='normal',
                 dim_feat=None):
        super().__init__()
        self.with_cp = with_cp              #
        self.query_norm = norm_layer(dim)
        if dim_feat is None:
            dim_feat = dim
        self.feat_norm = norm_layer(dim_feat)
        
        self.attn_type = attn_type
        if attn_type == 'normal':
            self.attn = CrossAttention(
                dim=dim, num_heads=num_heads, qkv_bias=False, 
                attn_drop=0., proj_drop=0.,
                dim_feat=dim_feat,
            )
        elif attn_type == 'deform':
            assert has_deform_attn
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.attn = MSDeformAttn(d_model=dim, 
                                         n_levels=n_levels, 
                                         n_heads=num_heads,
                                         n_points=n_points,
                                         ratio=deform_ratio,
                                         d_feat=dim_feat)
        else:
            raise NotImplementedError(f'Unknown attn_type {attn_type}')
        
        self.ca_gamma = nn.Parameter(0. * torch.ones((dim)), requires_grad=True)
        self.cffn_gamma = nn.Parameter(0. * torch.ones((dim)), requires_grad=True)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        else:
            self.cffn_gamma = nn.Identity()
    
    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W):
        
        def _inner_forward(query, feat):
            if self.attn_type == 'normal':
                attn = self.attn(self.query_norm(query), self.feat_norm(feat))
            elif self.attn_type == 'deform':
                dtype = query.dtype
                self.attn = self.attn.float()
                attn = self.attn(self.query_norm(query).float(), reference_points,
                                self.feat_norm(feat).float(), spatial_shapes,
                                level_start_index, None)
                attn = attn.to(dtype=dtype)

            query = query + self.ca_gamma * attn
            
            if self.with_cffn:
                query = query + self.cffn_gamma * self.drop_path(self.ffn(self.ffn_norm(query), H, W))
                
            return query
        
        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)
        
        return query



# branch1 和 branch2之间如何去进行交互？原来的是两个都交互。现在我只让branch1去给branch2交互。branch2不反馈branch1
class BidirectionalInteractionUnit(nn.Module):
    def __init__(self, branch1_dim, branch2_dim, 
                 branch1_feat_size, branch2_feat_size, 
                 num_heads=6, 
                 n_points=4, 
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop=0., 
                 drop_path=0., 
                 with_cffn=False, 
                 cffn_ratio=0.25, 
                 deform_ratio=1.0,
                 with_cp=False, 
                 attn_type='normal', 
                 with_proj=True):
        super().__init__()
        self.attn_type = attn_type
        self.branch1_feat_size = branch1_feat_size # only for calculating flops
        self.branch2_feat_size = branch2_feat_size
        self.branch1_dim = branch1_dim
        self.branch2_dim = branch2_dim
        
        self.with_proj = with_proj
        
        if with_proj:
            self.branch2to1_proj = nn.Linear(branch2_dim, branch1_dim)
            self.branch1to2_proj = nn.Linear(branch1_dim, branch2_dim)
            
        self.branch2to1_injector = Injector(dim=branch1_dim,
                                            num_heads=num_heads,
                                            n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
                                            with_cp=with_cp, with_cffn=with_cffn, cffn_ratio=cffn_ratio, drop=drop, 
                                            drop_path=drop_path,
                                            attn_type=attn_type,
                                            dim_feat=branch1_dim if with_proj else branch2_dim)
        
        self.branch1to2_injector = Injector(dim=branch2_dim,
                                            num_heads=num_heads,
                                            n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
                                            with_cp=with_cp, with_cffn=with_cffn, cffn_ratio=cffn_ratio, drop=drop, 
                                            drop_path=drop_path,
                                            attn_type=attn_type,
                                            dim_feat=branch2_dim if with_proj else branch1_dim)
        
    
    def forward(self, x1, x2, deform_inputs1, deform_inputs2, H1, W1, H2, W2):
        # x1 is small image (large model), x2 is large image (small model)
        
        if self.with_proj:
            x1_branch1to2_proj = self.branch1to2_proj(x1)
            x2_branch2to1_proj = self.branch2to1_proj(x2) 
        else:
            x1_branch1to2_proj = x1
            x2_branch2to1_proj = x2
            
        x1 = self.branch2to1_injector(query=x1, reference_points=deform_inputs1[0],
                                    feat=x2_branch2to1_proj, spatial_shapes=deform_inputs1[1],
                                    level_start_index=deform_inputs1[2], H=H1, W=W1)
        x2 = self.branch1to2_injector(query=x2, reference_points=deform_inputs2[0],
                                    feat=x1_branch1to2_proj, spatial_shapes=deform_inputs2[1],
                                    level_start_index=deform_inputs2[2], H=H2, W=W2) 
        return x1, x2


class BidirectionalInteractionUnit_atl(nn.Module):
    def __init__(self, 
                 branch1_dim, branch2_dim, 
                 branch1_feat_size, branch2_feat_size, 
                 num_heads=6, 
                 n_points=4, 
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop=0., 
                 drop_path=0., 
                 with_cffn=False, 
                 cffn_ratio=0.25, 
                 deform_ratio=1.0,
                 with_cp=False, 
                 attn_type='normal', 
                 with_proj=True):
        super().__init__()
        # import pdb;pdb.set_trace()
        self.attn_type = attn_type # deform
        self.branch1_feat_size = branch1_feat_size # only for calculating flops 128
        self.branch2_feat_size = branch2_feat_size #                            128
        self.branch1_dim = branch1_dim  # 128
        self.branch2_dim = branch2_dim  # 128
        
        self.with_proj = with_proj # True
        
        if with_proj:
            self.branch2to1_proj = nn.Linear(branch2_dim, branch1_dim)
            # self.branch1to2_proj = nn.Linear(branch1_dim, branch2_dim)
            
        self.branch2to1_injector = Injector(dim=branch1_dim,
                                            num_heads=num_heads,
                                            n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
                                            with_cp=with_cp, with_cffn=with_cffn, cffn_ratio=cffn_ratio, drop=drop, 
                                            drop_path=drop_path,
                                            attn_type=attn_type,
                                            dim_feat=branch1_dim if with_proj else branch2_dim)
        
        # self.branch1to2_injector = Injector(dim=branch2_dim,
        #                                     num_heads=num_heads,
        #                                     n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
        #                                     with_cp=with_cp, with_cffn=with_cffn, cffn_ratio=cffn_ratio, drop=drop, 
        #                                     drop_path=drop_path,
        #                                     attn_type=attn_type,
        #                                     dim_feat=branch2_dim if with_proj else branch1_dim)
        
    
    def forward(self, x1, x2, deform_inputs1, deform_inputs2, H1, W1, H2, W2):
        # x1 is small image (large model), x2 is large image (small model)
        
        if self.with_proj:
            # x1_branch1to2_proj = self.branch1to2_proj(x1)
            x2_branch2to1_proj = self.branch2to1_proj(x2) 
        else:
            # x1_branch1to2_proj = x1
            x2_branch2to1_proj = x2
        
        x1 = self.branch2to1_injector(query=x1, reference_points=deform_inputs1[0],
                                      feat=x2_branch2to1_proj, spatial_shapes=deform_inputs1[1],
                                      level_start_index=deform_inputs1[2], H=H1, W=W1)
        
        x2 = x2
        # x2 = self.branch1to2_injector(query=x2, reference_points=deform_inputs2[0],
        #                             feat=x1_branch1to2_proj, spatial_shapes=deform_inputs2[1],
        #                             level_start_index=deform_inputs2[2], H=H2, W=W2) 
        return x1, x2



def forward_blocks_atl(x, H, W, blocks, cls_=None):
    if len(blocks) == 0:
        print("!!! no blocks")
        return x, cls_
    
    if cls_ is not None:
        x = torch.cat((cls_, x), dim=1)
   
    if isinstance(blocks[0], ConvNextLayerWrapper):
        for _, blk in enumerate(blocks):
            x, H, W = blk(x, H, W)
    
    # 包含ConvNextBlock
    elif any(isinstance(b, ConvNeXtBlock) for b in blocks): 
        for _, blk in enumerate(blocks):
            # 分开处理下采样层和 convnext block
            if isinstance(blk, ConvNeXtBlock):
                bs, n, dim = x.shape # [1, 16384, 128]
                x = x.transpose(1,2).reshape(bs, dim, H, W) #[1, 16384, 128]->[1,128,128,128]
                x = blk(x) # [1, 128, 128, 128]
                bs, dim_new, H_new, W_new = x.shape
                x = x.reshape(bs, dim_new, H_new * W_new).transpose(1,2)
                H, W = H_new, W_new
            elif isinstance(blk, nn.Sequential): # 下采样层
                bs, n, dim = x.shape # [1, 16384, 128]
                x = x.transpose(1,2).reshape(bs, dim, H, W) #[1, 16384, 128]->[1,128,128,128]
                x = blk(x) # [1, 128, 128, 128] -> [1,256,64,64]
                bs, dim_new, H_new, W_new = x.shape
                x = x.reshape(bs, dim_new, H_new * W_new).transpose(1,2) # [1,256,64,64]->[1,4096,256]
                H, W = H_new, W_new
                # 下采样层

    else:
        for _, blk in enumerate(blocks):
            x = blk(x, H, W)
    
    if cls_ is not None:
        cls_, x = x[:, :1, :], x[:, 1:, :]
    
    return x, cls_, H, W


def forward_blocks(x, H, W, blocks, cls_=None):
    if len(blocks) == 0:
        print("!!! no blocks")
        return x, cls_
    
    if cls_ is not None:
        x = torch.cat((cls_, x), dim=1)
    
    if isinstance(blocks[0], ConvNextLayerWrapper):
        for _, blk in enumerate(blocks):
            x, H, W = blk(x, H, W)
    else:
        for _, blk in enumerate(blocks):
            x = blk(x, H, W)
    
    if cls_ is not None:
        cls_, x = x[:, :1, :], x[:, 1:, :]
    return x, cls_, H, W
   

class FourBranchInteractionBlock(nn.Module):
    def __init__(self, branch1_dim, branch2_dim, branch3_dim, branch4_dim, 
                 branch1_feat_size, branch2_feat_size, branch3_feat_size, branch4_feat_size,
                 attn_type='deform', **kwargs):
        super().__init__()
        self.attn_type = attn_type

        
        self.interaction_units_12 = BidirectionalInteractionUnit(branch1_dim, branch2_dim, branch1_feat_size, branch2_feat_size, attn_type=attn_type, **kwargs)
        self.interaction_units_23 = BidirectionalInteractionUnit(branch2_dim, branch3_dim, branch2_feat_size, branch3_feat_size, attn_type=attn_type, **kwargs)
        self.interaction_units_34 = BidirectionalInteractionUnit(branch3_dim, branch4_dim, branch3_feat_size, branch4_feat_size, attn_type=attn_type, **kwargs)
        
        # for calculating flops
        self.interaction_units = [
            self.interaction_units_12,
            self.interaction_units_23,
            self.interaction_units_34,
        ]
        
        self.branch1_dim = branch1_dim
        self.branch2_dim = branch2_dim
        self.branch3_dim = branch3_dim
        self.branch4_dim = branch4_dim
    
    def forward(self, x1, x2, x3, x4, branch1_blocks, branch2_blocks, branch3_blocks, branch4_blocks,
                H1, W1, H2, W2, H3, W3, H4, W4, deform_inputs=None, cls1=None, cls2=None, cls3=None, cls4=None):
        
        x1, cls1, H1, W1 = forward_blocks(x1, H1, W1, branch1_blocks, cls1)
        x2, cls2, H2, W2 = forward_blocks(x2, H2, W2, branch2_blocks, cls2)
        x3, cls3, H3, W3 = forward_blocks(x3, H3, W3, branch3_blocks, cls3)
        x4, cls4, H4, W4 = forward_blocks(x4, H4, W4, branch4_blocks, cls4)
        
        x3, x4 = self.interaction_units_34(x3, x4, deform_inputs["4to3"], deform_inputs["3to4"], H3, W3, H4, W4)
        x2, x3 = self.interaction_units_23(x2, x3, deform_inputs["3to2"], deform_inputs["2to3"], H2, W2, H3, W3)
        x1, x2 = self.interaction_units_12(x1, x2, deform_inputs["2to1"], deform_inputs["1to2"], H1, W1, H2, W2)
        
        return x1, x2, x3, x4, cls1, cls2, cls3, cls4



class ThreeBranchInteractionBlock(nn.Module):
    def __init__(self, 
                 branch1_dim, 
                 branch2_dim, 
                 branch3_dim, 
                 branch1_feat_size, 
                 branch2_feat_size, 
                 branch3_feat_size, 
                 attn_type='deform', **kwargs):
        super().__init__()
        self.attn_type = attn_type

        self.interaction_units_12 = BidirectionalInteractionUnit(branch1_dim, branch2_dim, branch1_feat_size, branch2_feat_size, attn_type=attn_type, **kwargs)
        self.interaction_units_23 = BidirectionalInteractionUnit(branch2_dim, branch3_dim, branch2_feat_size, branch3_feat_size, attn_type=attn_type, **kwargs)
        
        # for calculating flops
        self.interaction_units = [
            self.interaction_units_12,
            self.interaction_units_23,
        ]
        
        self.branch1_dim = branch1_dim
        self.branch2_dim = branch2_dim
        self.branch3_dim = branch3_dim
    
    def forward(self, x1, x2, x3, 
                branch1_blocks, branch2_blocks, branch3_blocks, 
                H1, W1, H2, W2, H3, W3, 
                deform_inputs=None, 
                cls1=None, cls2=None, cls3=None):
        x1, cls1, H1, W1 = forward_blocks(x1, H1, W1, branch1_blocks, cls1)
        x2, cls2, H2, W2 = forward_blocks(x2, H2, W2, branch2_blocks, cls2)
        x3, cls3, H3, W3 = forward_blocks(x3, H3, W3, branch3_blocks, cls3)
        
        x2, x3 = self.interaction_units_23(x2, x3, deform_inputs["3to2"], deform_inputs["2to3"], H2, W2, H3, W3)
        x1, x2 = self.interaction_units_12(x1, x2, deform_inputs["2to1"], deform_inputs["1to2"], H1, W1, H2, W2)
        
        return x1, x2, x3, cls1, cls2, cls3, H1, W1, H2, W2, H3, W3





class TwoBranchInteractionBlock(nn.Module):
    def __init__(self, 
                 branch1_dim, 
                 branch2_dim, 
                 branch1_feat_size, 
                 branch2_feat_size, 
                 attn_type='deform', **kwargs):
        super().__init__()
        self.attn_type = attn_type
                                     # 只有branch2 给 branch1 
        self.interaction_units_12 = BidirectionalInteractionUnit_atl(branch1_dim, branch2_dim, branch1_feat_size, branch2_feat_size, attn_type=attn_type, **kwargs)
        
        # for calculating flops
        self.interaction_units = [
            self.interaction_units_12,
        ]
        
        self.branch1_dim = branch1_dim
        self.branch2_dim = branch2_dim
    
    def forward(self, x1, x2, 
                branch1_blocks, branch2_blocks, 
                H1, W1, H2, W2, 
                deform_inputs=None, 
                cls1=None, cls2=None):
        x1, cls1, H1, W1 = forward_blocks_atl(x1, H1, W1, branch1_blocks, cls1)  # x1过block
        x2, cls2, H2, W2 = forward_blocks_atl(x2, H2, W2, branch2_blocks, cls2)  # x2过block 

        x1, x2 = self.interaction_units_12(x1, x2, deform_inputs["2to1"], deform_inputs["1to2"], H1, W1, H2, W2)
        
        return x1, x2, cls1, cls2, H1, W1, H2, W2

# ============================= For Segnext =================================
class BidirectionalInteractionUnit_segnext(nn.Module):
    """先明确，这里是什么去交互呢？
    对于ViT来说，是[2,576,1024] 和 [2,1024,768]这两个维度去交互。
    对于Segnext来说，则是 [B, 25600, 64] [B, 16384, 64] 这两个维度去交互。
                         [B, 6400, 128] [B, 4096, 128]
                         [B, 1600, 320] [B, 1024, 320]
                         [B, 400, 512]  [B, 256,  512]
    
    交互完再reshape成2D图，x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
    然后交互前，再x, H, W = patch_embed(x), 然后交互
    """

    def __init__(self, 
                 branch1_dim,  # embed_dims[0] / embed_dims[1] [64, 128, 320, 512]
                 branch2_dim,  # [64, 128, 320, 512]
                 branch1_img_size,  # real_size 640 
                 branch2_img_size,  # real_size 640
                 num_heads=6, 
                 n_points=4, 
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop=0., 
                 drop_path=0., 
                 with_cffn=False, 
                 cffn_ratio=0.25, 
                 deform_ratio=1.0, 
                 with_cp=False, 
                 attn_type='normal', 
                 with_proj=True):
        super().__init__()
        self.attn_type = attn_type
        self.branch1_img_size = branch1_img_size
        self.branch2_img_size = branch2_img_size
        self.branch1_dim = branch1_dim
        self.branch2_dim = branch2_dim
        
        self.with_proj = with_proj
        
        if with_proj:
            self.branch2to1_proj = nn.Linear(branch2_dim, branch1_dim) # 简单的变换维度
            # self.branch1to2_proj = nn.Linear(branch1_dim, branch2_dim) # 简单的变换维度
        
        # 2to1, 最后输出x1的特征,小的模型给大的模型
        # 输入的维度是1的维度
        # 中间特征的维度是2的维度（dim_feat=branch1_dim if with_proj else branch2_dim）
        # import pdb;pdb.set_trace()
        self.branch2to1_injector = Injector(dim=branch1_dim,          # 64
                                            num_heads=num_heads,      # 16
                                            n_points=n_points,        # 4
                                            norm_layer=norm_layer,    # 
                                            deform_ratio=deform_ratio,# 0.5
                                            with_cp=with_cp,          # False
                                            with_cffn=with_cffn,      # True
                                            cffn_ratio=cffn_ratio,    # 0.25
                                            drop=drop,                # 0.0 
                                            drop_path=drop_path,      # 0.4
                                            attn_type=attn_type,      # 'deform '
                                            dim_feat=branch1_dim if with_proj else branch2_dim)  # 64
        
        # 1to2, 最后输出x2的特征 大的模型给小的模型
        self.branch1to2_injector = Injector(dim=branch2_dim,
                                            num_heads=num_heads,
                                            n_points=n_points, 
                                            norm_layer=norm_layer, 
                                            deform_ratio=deform_ratio,
                                            with_cp=with_cp, 
                                            with_cffn=with_cffn, 
                                            cffn_ratio=cffn_ratio, 
                                            drop=drop, 
                                            drop_path=drop_path,
                                            attn_type=attn_type,
                                            dim_feat=branch2_dim if with_proj else branch1_dim)
        
    
    def forward(self, x1, x2, deform_inputs1, deform_inputs2, H1, W1, H2, W2):
        # x1 is small image (large model), x2 is large image (small model)
        
        if self.with_proj:
            # x1_branch1to2_proj = self.branch1to2_proj(x1) # 简单的变换维度
            x2_branch2to1_proj = self.branch2to1_proj(x2) # 简单的变换维度
        else:
            # x1_branch1to2_proj = x1
            x2_branch2to1_proj = x2
        # import pdb;pdb.set_trace()                                          # base的
        x1 = self.branch2to1_injector(query=x1,                             # x1[2, 16384, 64]
                                      reference_points=deform_inputs1[0],   #   [1, 16384, 1,2]
                                      feat=x2_branch2to1_proj,              # x2[2,25600,64]
                                      spatial_shapes=deform_inputs1[1],     # [160,160]
                                      level_start_index=deform_inputs1[2],  # [0]
                                      H=H1,                                 # 128
                                      W=W1)                                 # 128
        
        # x2 = self.branch1to2_injector(query=x2,                             # x2[2,25600,64]
        #                               reference_points=deform_inputs2[0],   # [1,25600,1,2]
        #                               feat=x1_branch1to2_proj,              # x1[2,16384,64]  
        #                               spatial_shapes=deform_inputs2[1],     # [128,128]
        #                               level_start_index=deform_inputs2[2],  # 0  
        #                               H=H2,                                 # 160
        #                               W=W2)                                 # 160
        return x1, x2

class ThreeBranchInteractionBlock_segnext(nn.Module):
    def __init__(self, 
                 branch1_dim, 
                 branch2_dim, 
                 branch3_dim, 

                 branch1_img_size, 
                 branch2_img_size, 
                 branch3_img_size, 
                 attn_type='deform', 
                 **kwargs):
        super().__init__()
        self.attn_type = attn_type

        self.interaction_units_12 = BidirectionalInteractionUnit_segnext(branch1_dim, branch2_dim, branch1_img_size, branch2_img_size, attn_type=attn_type, **kwargs)
        self.interaction_units_23 = BidirectionalInteractionUnit_segnext(branch2_dim, branch3_dim, branch2_img_size, branch3_img_size, attn_type=attn_type, **kwargs)
        
        # for calculating flops
        self.interaction_units = [
            self.interaction_units_12,
            self.interaction_units_23,
        ]
        
        self.branch1_dim = branch1_dim
        self.branch2_dim = branch2_dim
        self.branch3_dim = branch3_dim
    
    def forward_vit_blocks(self, x, H, W, blocks, cls_=None):
        if cls_ is not None:
            x = torch.cat((cls_, x), dim=1)
        for _, blk in enumerate(blocks):
            x = blk(x, H, W)
        if cls_ is not None:
            cls_, x = x[:, :1, :], x[:, 1:, :]
        return x, cls_

    def forward(self, 
                x1, x2, x3, 
                H1, W1, H2, W2, H3, W3,
                deform_inputs=None):
        
        # import pdb;pdb.set_trace()
        # 特征交互完的特征
        # deform传给了 BidirectionalInteractionUnit_segnext的forward
        # 这里给的x 需要是一维向量的形式，需要在这,把x变成 从[2,64,160,160]-->[2,25600,64]
        x2, x3 = self.interaction_units_23(x2, x3, deform_inputs["3to2"], deform_inputs["2to3"], H2, W2, H3, W3)
        x1, x2 = self.interaction_units_12(x1, x2, deform_inputs["2to1"], deform_inputs["1to2"], H1, W1, H2, W2)
        
        return x1, x2, x3
