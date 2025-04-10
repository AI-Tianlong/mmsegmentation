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
from mmseg.models.backbones.deit3 import DeiT3
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
    from ..._base_.datasets.S2_5B_18class_512 import *
    from ..._base_.default_runtime import *
    from ..._base_.schedules.schedule_80k import *


# 通过os 设置 TORCH_DISTRIBUTED_DEBUG 为 INFO
# import os 
# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
# find_unused_parameters = True

num_classes = 18
norm_cfg = dict(type=SyncBN, requires_grad=True) # decode_head的 norm_cfg
pretrained = 'checkpoints/2-对比实验的权重/deit3/10chan/deit3-large-384px-10chan.pth'

crop_size = (640, 640)
data_preprocessor = dict(
    type=SegDataPreProcessor,
    mean = [412.62603765, 317.66892688, 243.74720123, 292.61469172],
    std = [42.79585263, 45.59081086, 54.94280476, 69.32133677],
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

model = dict(
    type=EncoderDecoder,
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type=DeiT3,
        arch='large',
        in_channels=4,
        img_size=crop_size[0],
        patch_size=16,
        drop_path_rate=0.4,
        out_type='featmap',
        out_indices=(7, 11, 15, 23), # -1 ?测试一下
        init_cfg=dict(type='Pretrained', checkpoint=pretrained, prefix='backbone.'),
        ),
    neck=dict(
        type=MultiLevelNeck,
        in_channels=[1024, 1024, 1024, 1024],
        out_channels=1024,
        scales=[4, 2, 1, 0.5]),
    decode_head=dict(
        type=UPerHead,
        in_channels=[1024, 1024, 1024, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=1024,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0)
    ),
    test_cfg=dict(mode='whole')                                       
)

# 和 vit vit_deit的配置一样
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(
        type=AdamW, lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

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
train_cfg = dict(type=IterBasedTrainLoop, max_iters=80000, val_interval=8000)
val_cfg = dict(type=ValLoop)
test_cfg = dict(type=TestLoop)

default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(type=CheckpointHook, by_epoch=False, interval=2000, max_keep_ckpts=2),
    sampler_seed=dict(type=DistSamplerSeedHook),
    visualization=dict(type=SegVisualizationHook))

val_evaluator = dict(
    type=IoUMetric, iou_metrics=['mIoU', 'mFscore'])  # 'mDice', 'mFscore'
test_evaluator = dict(
    type=IoUMetric,
    iou_metrics=['mIoU', 'mFscore'],
    # format_only=True,
    keep_results=True)
