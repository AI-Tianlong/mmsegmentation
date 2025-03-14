# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------


from mmengine.config import read_base
from mmengine.optim.optimizer import OptimWrapper
from mmengine.optim.scheduler.lr_scheduler import LinearLR, PolyLR
from torch.nn.modules.batchnorm import SyncBatchNorm as SyncBN
from torch.optim import AdamW


from torch.nn.modules.activation import GELU
from torch.nn.modules.batchnorm import SyncBatchNorm as SyncBN
from torch.nn.modules.normalization import GroupNorm as GN

# EncoderDecoder
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from mmseg.models.segmentors.atl_hiera_37_encoder_decoder import ATL_Hiera_EncoderDecoder
# SegDataPreProcessor
from mmseg.models.data_preprocessor import SegDataPreProcessor
# Backbone
from mmseg.models.backbones.mscan import MSCAN
from mmseg.models.backbones.piip_3branch_segnext import PIIPThreeBranch_conv
# from mmseg.models.backbones.piip_2branch import PIIPTwoBranch
# from mmseg.models.backbones.piip_3branch import PIIPThreeBranch
# Neck
from mmseg.models.necks.multilevel_neck import  MultiLevelNeck
# DecodeHead
from mmseg.models.decode_heads.ham_head import LightHamHead
from mmseg.models.decode_heads.uper_head import UPerHead
from mmseg.models.decode_heads.fcn_head import FCNHead
# Loss
from mmseg.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmseg.models.losses.atl_hiera_37_loss_convseg import ATL_Hiera_Loss_convseg
#optimizer
from mmseg.engine.optimizers.piip_layer_decay_optimizer_constructor import CustomLayerDecayOptimizerConstructor

# Evaluation
from mmseg.evaluation import IoUMetric


with read_base():
    from ..._base_.datasets.a_atl_0_paper_5b_GF2_18class_224 import *
    from ..._base_.default_runtime import *
    from ..._base_.schedules.schedule_80k import *

branch1_large_pretrained = 'checkpoints/2-对比实验的权重/segnext/large/segnext_mscan_l_4chan.pth'
branch2_base_pretrained  = 'checkpoints/2-对比实验的权重/segnext/base/segnext_mscan_b_4chan.pth'
branch3_small_pretrained = 'checkpoints/2-对比实验的权重/segnext/small/segnext_mscan_s_4chan.pth'


ham_norm_cfg = dict(type=GN, num_groups=32, requires_grad=True)


norm_cfg = dict(type=SyncBN, requires_grad=True)
num_classes = 18

crop_size = (640, 640)
data_preprocessor = dict(
    type=SegDataPreProcessor,
    mean =[454.1608733420, 320.6480230485 , 238.9676917808 , 301.4478970428],
    std =[55.4731833972, 51.5171917858, 62.3875607521, 82.6082214602],
    # bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
    test_cfg=dict(size_divisor=32))

model = dict(
    type=EncoderDecoder,
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type=PIIPThreeBranch_conv,
        n_points=4,
        deform_num_heads=16,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        with_cffn=True,
        interact_attn_type='deform',  # 可以换成别的注意力！
        interaction_drop_path_rate=0.4,
        interaction_proj=False,
        norm_layer='none',
        # ViT-large
        
        branch1=dict(
            in_channels=4,
            real_size=640,   # 该分支图像的尺寸
            pretrain_img_size=224,
            embed_dims=[64, 128, 320, 512], # 四个stage的输出特征图维度 # ViT类的这个stage不变啊！所以这个不行
            mlp_ratios=[8, 8, 4, 4],
            drop_rate=0.0,
            depths=[3, 5, 27, 3], # 每一个stage 包含的MSCA模块的数量？
            attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
            attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
            act_cfg=dict(type=GELU),
            norm_cfg=dict(type=SyncBN, requires_grad=True),
            interaction_indexes=[[0, 0], [1, 1], [2, 2], [3, 3]], # 总共就四个stage,只能这么交互
            pretrained = branch1_large_pretrained,
        ),
        # ViT-base
        branch2=dict(
            pretrained = branch2_base_pretrained, 
            in_channels=4,
            real_size=640,
            pretrain_img_size=224,
            embed_dims=[64, 128, 320, 512],
            mlp_ratios=[8, 8, 4, 4],
            drop_rate=0.0,
            drop_path_rate=0.1,
            depths=[3, 3, 12, 3],
            attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
            attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
            act_cfg=dict(type=GELU),
            norm_cfg=dict(type=SyncBN, requires_grad=True),
            interaction_indexes=[[0, 0], [1, 1], [2, 2], [3, 3]], # 总共就四个stage,只能这么交互
        ),
        # ViT-small
        branch3=dict(
            pretrained = branch3_small_pretrained,
            in_channels=4,
            real_size=640,
            pretrain_img_size=224,
            embed_dims=[64, 128, 320, 512],
            mlp_ratios=[8, 8, 4, 4],
            drop_rate=0.0,
            drop_path_rate=0.1,
            depths=[2, 2, 4, 2],
            attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
            attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
            act_cfg=dict(type=GELU),
            norm_cfg=dict(type=SyncBN, requires_grad=True),
            interaction_indexes=[[0, 0], [1, 1], [2, 2], [3, 3]], # 总共就四个stage,只能这么交互
        ),
    ),
    # neck=dict(
    #     type=MultiLevelNeck,
    #     in_channels=[1024, 1024, 1024, 1024],
    #     out_channels=1024,
    #     scales=[4, 2, 1, 0.5]),
    decode_head=dict(
        type=LightHamHead,
        in_channels=[128, 320, 512],
        in_index=[1, 2, 3],
        channels=1024,
        ham_channels=1024,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=ham_norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0),
        ham_kwargs=dict(
            MD_S=1,
            MD_R=16,
            train_steps=6,
            eval_steps=7,
            inv_t=100,
            rand_init=True)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))


# optimizer
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(
        type=AdamW, lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=optimizer,
    constructor=CustomLayerDecayOptimizerConstructor,
    paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.85, skip_stride=[2, 2]))

param_scheduler = [
    dict(
        type=LinearLR, start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type=PolyLR,
        power=1.0,
        begin=1500,
        end=80000,
        eta_min=0.0,
        by_epoch=False,
    )
]


train_cfg.update(type=IterBasedTrainLoop, max_iters=80000, val_interval=4000)
default_hooks.update(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(type=CheckpointHook, by_epoch=False, interval=2000, max_keep_ckpts=3),
    sampler_seed=dict(type=DistSamplerSeedHook),
    visualization=dict(type=SegVisualizationHook))


val_evaluator = dict(
    type=IoUMetric, iou_metrics=['mIoU', 'mFscore'])  # 'mDice', 'mFscore'
test_evaluator = dict(
    type=IoUMetric,
    iou_metrics=['mIoU', 'mFscore'],
    # format_only=True,
    keep_results=True)