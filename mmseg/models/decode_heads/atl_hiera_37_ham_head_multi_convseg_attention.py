# Copyright (c) OpenMMLab. All rights reserved.
# Originally from https://github.com/visual-attention-network/segnext
# Licensed under the Apache License, Version 2.0 (the "License")
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.device import get_device

from mmseg.registry import MODELS
from ..utils import resize
from .decode_head import BaseDecodeHead

from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule, build_norm_layer
from mmseg.models.losses import accuracy
from mmseg.registry import MODELS
from mmseg.utils import SampleList, ConfigType
from typing import List, Tuple
from torch import Tensor

class Matrix_Decomposition_2D_Base(nn.Module):
    """Base class of 2D Matrix Decomposition.

    Args:
        MD_S (int): The number of spatial coefficient in
            Matrix Decomposition, it may be used for calculation
            of the number of latent dimension D in Matrix
            Decomposition. Defaults: 1.
        MD_R (int): The number of latent dimension R in
            Matrix Decomposition. Defaults: 64.
        train_steps (int): The number of iteration steps in
            Multiplicative Update (MU) rule to solve Non-negative
            Matrix Factorization (NMF) in training. Defaults: 6.
        eval_steps (int): The number of iteration steps in
            Multiplicative Update (MU) rule to solve Non-negative
            Matrix Factorization (NMF) in evaluation. Defaults: 7.
        inv_t (int): Inverted multiple number to make coefficient
            smaller in softmax. Defaults: 100.
        rand_init (bool): Whether to initialize randomly.
            Defaults: True.
    """

    def __init__(self,
                 MD_S=1,
                 MD_R=64,
                 train_steps=6,
                 eval_steps=7,
                 inv_t=100,
                 rand_init=True):
        super().__init__()

        self.S = MD_S
        self.R = MD_R

        self.train_steps = train_steps
        self.eval_steps = eval_steps

        self.inv_t = inv_t

        self.rand_init = rand_init

    def _build_bases(self, B, S, D, R, device=None):
        raise NotImplementedError

    def local_step(self, x, bases, coef):
        raise NotImplementedError

    def local_inference(self, x, bases):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    def forward(self, x, return_bases=False):
        """Forward Function."""
        B, C, H, W = x.shape

        # (B, C, H, W) -> (B * S, D, N)
        D = C // self.S
        N = H * W
        x = x.view(B * self.S, D, N)
        if not self.rand_init and not hasattr(self, 'bases'):
            bases = self._build_bases(1, self.S, D, self.R, device=x.device)
            self.register_buffer('bases', bases)

        # (S, D, R) -> (B * S, D, R)
        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R, device=x.device)
        else:
            bases = self.bases.repeat(B, 1, 1)

        bases, coef = self.local_inference(x, bases)

        # (B * S, N, R)
        coef = self.compute_coef(x, bases, coef)

        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = torch.bmm(bases, coef.transpose(1, 2))

        # (B * S, D, N) -> (B, C, H, W)
        x = x.view(B, C, H, W)

        return x


class NMF2D(Matrix_Decomposition_2D_Base):
    """Non-negative Matrix Factorization (NMF) module.

    It is inherited from ``Matrix_Decomposition_2D_Base`` module.
    """

    def __init__(self, args=dict()):
        super().__init__(**args)

        self.inv_t = 1

    def _build_bases(self, B, S, D, R, device=None):
        """Build bases in initialization."""
        if device is None:
            device = get_device()
        bases = torch.rand((B * S, D, R)).to(device)
        bases = F.normalize(bases, dim=1)

        return bases

    def local_step(self, x, bases, coef):
        """Local step in iteration to renew bases and coefficient."""
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ [(B * S, D, R)^T @ (B * S, D, R)] -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # Multiplicative Update
        coef = coef * numerator / (denominator + 1e-6)

        # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
        numerator = torch.bmm(x, coef)
        # (B * S, D, R) @ [(B * S, N, R)^T @ (B * S, N, R)] -> (B * S, D, R)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        # Multiplicative Update
        bases = bases * numerator / (denominator + 1e-6)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        """Compute coefficient."""
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ (B * S, D, R)^T @ (B * S, D, R) -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # multiplication update
        coef = coef * numerator / (denominator + 1e-6)

        return coef


class Hamburger(nn.Module):
    """Hamburger Module. It consists of one slice of "ham" (matrix
    decomposition) and two slices of "bread" (linear transformation).

    Args:
        ham_channels (int): Input and output channels of feature.
        ham_kwargs (dict): Config of matrix decomposition module.
        norm_cfg (dict | None): Config of norm layers.
    """

    def __init__(self,
                 ham_channels=512,
                 ham_kwargs=dict(),
                 norm_cfg=None,
                 **kwargs):
        super().__init__()

        self.ham_in = ConvModule(
            ham_channels, ham_channels, 1, norm_cfg=None, act_cfg=None)

        self.ham = NMF2D(ham_kwargs)

        self.ham_out = ConvModule(
            ham_channels, ham_channels, 1, norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, x):
        enjoy = self.ham_in(x)
        enjoy = F.relu(enjoy, inplace=True)
        enjoy = self.ham(enjoy)
        enjoy = self.ham_out(enjoy)
        ham = F.relu(x + enjoy, inplace=True)

        return ham



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
class ATL_Hiera_LightHamHead_Multi_convseg_attentation(BaseDecodeHead):
    """SegNeXt decode head.

    This decode head is the implementation of `SegNeXt: Rethinking
    Convolutional Attention Design for Semantic
    Segmentation <https://arxiv.org/abs/2209.08575>`_.
    Inspiration from https://github.com/visual-attention-network/segnext.

    Specifically, LightHamHead is inspired by HamNet from
    `Is Attention Better Than Matrix Decomposition?
    <https://arxiv.org/abs/2109.04553>`.

    Args:
        ham_channels (int): input channels for Hamburger.
            Defaults: 512.
        ham_kwargs (int): kwagrs for Ham. Defaults: dict().
    """

    def __init__(self, 
                 ham_channels=512, 
                 ham_kwargs=dict(),
                 num_classes_level_list=[5,10,19],
                 merge_hiera: bool = True,
                 proj: str = 'convmlp',
                 **kwargs):
        
        num_classes = num_classes_level_list[-1] # 去创建 self.conv_seg
        super().__init__(num_classes=num_classes,
                         input_transform='multiple_select', 
                         **kwargs)
        
        self.ham_channels = ham_channels

        # ====================== Hiera ======================
        self.merge_hiera = merge_hiera
        
        if isinstance(num_classes_level_list, list):
            self.num_classes_level_list = num_classes_level_list  # [5,9,10]

            self.conv_seg_L1 = nn.Conv2d(self.channels, num_classes_level_list[0], kernel_size=1) #(1024-->5)
            self.conv_attention_L1 = nn.Conv2d(num_classes_level_list[0], 1, kernel_size=1) #(5-->1)
    
            self.conv_seg_L2 = nn.Conv2d(self.channels, num_classes_level_list[1], kernel_size=1)
            self.conv_attention_L2 = nn.Conv2d(num_classes_level_list[1], 1, kernel_size=1) #(10-->1)

            self.conv_seg_L3 = self.conv_seg #(1024-->12)



        elif isinstance(num_classes_level_list, int):
            num_classes = num_classes

        self.proj_head = ProjectionHead(dim_in=512,   # 2048-->256 # backbone的最后一个输出
                                        norm_cfg=self.norm_cfg, 
                                        proj=proj)
        self.register_buffer('step', torch.zeros(1))
        # ====================== Hiera ======================

        self.squeeze = ConvModule(
            sum(self.in_channels),
            self.ham_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.hamburger = Hamburger(ham_channels, ham_kwargs, **kwargs)

        self.align = ConvModule(
            self.ham_channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)


    # def forward(self, inputs):
    #     """Forward function."""
    #     inputs = self._transform_inputs(inputs)

    #     inputs = [
    #         resize(
    #             level,
    #             size=inputs[0].shape[2:],
    #             mode='bilinear',
    #             align_corners=self.align_corners) for level in inputs
    #     ]

    #     inputs = torch.cat(inputs, dim=1)
    #     # apply a conv block to squeeze feature map
    #     x = self.squeeze(inputs)
    #     # apply hamburger module
    #     x = self.hamburger(x)

    #     # apply a conv block to align feature map
    #     output = self.align(x)
    #     output = self.cls_seg(output)
    #     return output

# ================================== Hiera ===================================
    # 分开输出的话
    def cls_seg(self, feat, conv_seg):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = conv_seg(feat)
        return output
    
    def forward(self, inputs):
        """Forward function."""
        # [[4, 64, 128, 128], [4, 128, 64, 64],[4, 320, 32, 32],[4, 512, 16, 16]]
        # in_index = [1,2,3] 
        embedding = self.proj_head(inputs[-1])  # [4,512,16,16]-->[4,256,16,16]
        # import pdb; pdb.set_trace()
        inputs = self._transform_inputs(inputs)
        # [[4, 128, 64, 64],[4, 320, 32, 32],[4, 512, 16, 16]]
        inputs = [
            resize(
                level,
                size=inputs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners) for level in inputs
        ]
        # [[4, 128, 64, 64],[4, 320, 64, 64],[4, 512, 64, 64]]
        # import pdb; pdb.set_trace()
        inputs = torch.cat(inputs, dim=1)  # [4,960,64,64]
        # apply a conv block to squeeze feature map
        x = self.squeeze(inputs)  # [4,1024,64,64]
        # apply hamburger module
        x = self.hamburger(x)     # [4,1024,64,64]

        # apply a conv block to align feature map
        output = self.align(x) # [4, 1024, 64, 64] --> [4, 1024, 64, 64]
        

        output_L1 = self.cls_seg(output, self.conv_seg_L1) # [4, 1024, 64, 64] --> [4, 5, 64, 64]
        output_L1_attentation = self.conv_attention_L1(output_L1) # [4, 5, 64, 64] --> [4, 1, 64, 64]
        output = output + output_L1_attentation # [4, 1024, 64, 64] --> [4, 1024, 64, 64] add L1特征图


        output_L2 = self.cls_seg(output, self.conv_seg_L2) # [4, 1024, 64, 64] --> [4, 10, 64, 64]
        output_L2_attentation = self.conv_attention_L2(output_L2) # [4, 10, 64, 64] --> [4, 1, 64, 64]
        output = output + output_L2_attentation # [4, 1024, 64, 64] --> [4, 1024, 64, 64] add L2特征图

        output_L3 = self.cls_seg(output, self.conv_seg) # [4, 1024, 64, 64] --> [4, 19, 64, 64]
        
        output_list = [output_L1, output_L2, output_L3]
        self.step += 1

        return output_list, embedding


    def loss_by_feat(self, seg_logits: Tensor,
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
        
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None
            # print('走的这里')

        seg_label = seg_label.squeeze(1)  # [2,1,512,512]-->[2,512,512]

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        
        # 如果loss_decode的config是多个dict的话，则构造的时候，self.loss_decode.append(MODELS.build(loss))
        # 否则只是 self.loss_decode = MODELS.build(loss_decode)
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:  # loss['atl_loss_ce'],log就打印decode.atl_loss_ce
                # pdb.set_trace()

                # loss['loss_hiera_ce'] = celoss()
                # loss['loss_hiera_treemin'] = treemin()
                # loss['loss_hiera_treetriplet'] = treetriplet()

                # import pdb; pdb.set_trace()
                loss[loss_decode.loss_name] = loss_decode(
                    self.step,
                    embedding,
                    seg_logits,
                    seg_label,
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
                # pdb.set_trace()
        # pdb.set_trace()

        if isinstance(seg_logits, list) and len(seg_logits) == 3:
            seg_logits = seg_logits[2]
        elif isinstance(seg_logits, torch.Tensor) and seg_logits.shape[1]==sum(self.num_classes_level_list):
            seg_logits = seg_logits[:,-self.num_classes_level_list[-1]:,:,:]

        loss['acc_seg'] = accuracy(
            seg_logits, seg_label, ignore_index=self.ignore_index)
        # pdb.set_trace()
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
                seg_logits, embedding = seg_logits

                if isinstance(seg_logits, list) and len(seg_logits) == 3:
                    # 仅输出L3作为结果
                    seg_logits = seg_logits[2]




                    assert seg_logits.shape[1] == self.num_classes_level_list[-1]
                elif isinstance(seg_logits, torch.Tensor) and seg_logits.shape[1]==sum(self.num_classes_level_list):
                    seg_logits = seg_logits[:,-self.num_classes_level_list[-1]:,:,:]
                    assert seg_logits.shape[1] == self.num_classes_level_list[-1]

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
