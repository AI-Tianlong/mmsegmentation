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
from mmseg.models.backbones.piip_2branch import PIIPTwoBranch
from mmseg.models.backbones.piip_3branch_piip_final import PIIPThreeBranch
from mmseg.models.backbones.internvit_6b import InternViT6B
# Neck
from mmseg.models.necks.multilevel_neck import  MultiLevelNeck
# DecodeHead
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
    from ..._base_.datasets.Google_5B_18class_896 import *
    from ..._base_.default_runtime import *
    from ..._base_.schedules.schedule_80k import *


norm_cfg = dict(type=SyncBN, requires_grad=True)
num_classes = 18

branch1_pretrained = '/data/AI-Tianlong/openmmlab/mmsegmentation/checkpoints/2-对比实验的权重/convnext-facebook/convnext-base-224'
branch2_pretrained = '/data/AI-Tianlong/openmmlab/mmsegmentation/checkpoints/2-对比实验的权重/convnext-facebook/convnext-small-224'
branch3_pretrained = '/data/AI-Tianlong/openmmlab/mmsegmentation/checkpoints/2-对比实验的权重/convnext-facebook/convnext-tiny-224'

# branch1_pretrained = 'checkpoints/2-对比实验的权重/convnext-facebook/convnext_tiny_22k_1k_224.pth'
# branch2_pretrained = 'checkpoints/2-对比实验的权重/convnext-facebook/convnext_small_22k_1k_224.pth'
# branch3_pretrained = 'checkpoints/2-对比实验的权重/convnext-facebook/convnext_base_22k_1k_224.pth'

crop_size = (896, 896)
data_preprocessor = dict(
    type=SegDataPreProcessor,
        mean = [123.675, 116.28, 103.53],
        std = [58.395, 57.12, 57.375],
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
    test_cfg=dict(size_divisor=32))

model = dict(
    type=EncoderDecoder,
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type=PIIPThreeBranch,
        n_points=4,
        deform_num_heads=16,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        with_cffn=True,
        interact_attn_type='deform',  # 'deform' or 'normal'
        interaction_drop_path_rate=0.4,
        with_simple_fpn=False,
        out_interaction_indexes=[0, 1, 10],

        # convnext-base
        branch1=dict(
            real_size=448, 
            interaction_indexes=[
                [0, 2], 
                [3, 6], 
                [7, 10], [11, 13], [14, 16], [17, 19], [20, 22], [23, 25], [26, 28], [29, 31], [32, 34],
                [35, 38],
            ],
            downsample_ratios=[
                4, 4, 4,
                8, # downsampling layer
                8, 8, 8, 
                16, # downsampling layer
                16, 16, 16, 16, 16, 16, 16, 16, 16,
                16, 16, 16, 16, 16, 16, 16, 16, 16,
                16, 16, 16, 16, 16, 16, 16, 16, 16,
                32, # downsampling layer
                32, 32, 32,
            ],
            pretrained=branch1_pretrained, # out_dim=2048
            drop_path_rate=0.4,
        ),
        
        # convnext-s
        branch2=dict(
            real_size=672,
            interaction_indexes=[
                [0, 2], 
                [3, 6], 
                [7, 10], [11, 13], [14, 16], [17, 19], [20, 22], [23, 25], [26, 28], [29, 31], [32, 34],
                [35, 38],
            ],
            downsample_ratios=[
                4, 4, 4,
                8, # downsampling layer
                8, 8, 8, 
                16, # downsampling layer
                16, 16, 16, 16, 16, 16, 16, 16, 16,
                16, 16, 16, 16, 16, 16, 16, 16, 16,
                16, 16, 16, 16, 16, 16, 16, 16, 16,
                32, # downsampling layer
                32, 32, 32,
            ],
            pretrained=branch2_pretrained, # out_dim=512
            drop_path_rate=0.3,
        ),
        
        # convnext-t
        branch3=dict(
            real_size=896,
            interaction_indexes=[
                [0, 2], 
                [3, 6], 
                [7, 8], [9, 9], [10, 10], [11, 11], [12, 12], [13, 13], [14, 14], [15, 15], [16, 16],
                [17, 20],
            ],
            downsample_ratios=[
                4, 4, 4,
                8, # downsampling layer
                8, 8, 8, 
                16, # downsampling layer
                16, 16, 16, 16, 16, 16, 16, 16, 16,
                32, # downsampling layer
                32, 32, 32,
            ],
            pretrained=branch3_pretrained, # out_dim=512
            drop_path_rate=0.3,
        ),
    ),
    # neck=dict(
    #     type=MultiLevelNeck,
    #     in_channels=[1024, 1024, 1024, 1024],
    #     out_channels=1024,
    #     scales=[4, 2, 1, 0.5]),

    decode_head=dict(
        type=UPerHead,
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=768,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    # test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341)))
    test_cfg=dict(mode='whole'))


optimizer = dict(
    type=AdamW,
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
)

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


# training schedule for 80k
train_cfg = dict(type=IterBasedTrainLoop, max_iters=80000, val_interval=4000)
val_cfg = dict(type=ValLoop)
test_cfg = dict(type=TestLoop)

default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(type=CheckpointHook, by_epoch=False, interval=2000, max_keep_ckpts=10),
    sampler_seed=dict(type=DistSamplerSeedHook),
    visualization=dict(type=SegVisualizationHook))


val_evaluator = dict(
    type=IoUMetric, iou_metrics=['mIoU', 'mFscore'])  # 'mDice', 'mFscore'
test_evaluator = dict(
    type=IoUMetric,
    iou_metrics=['mIoU', 'mFscore'],
    # format_only=True,
    keep_results=True)

