# 2024-09-02 测试, 可以跑通,loss-从5开始降低。

from mmcv.transforms import (LoadImageFromFile, RandomChoice,
                             RandomChoiceResize, RandomFlip)
from mmengine.config import read_base
from mmengine.optim.optimizer import OptimWrapper
from mmengine.optim.scheduler.lr_scheduler import LinearLR, PolyLR
from torch.nn.modules.batchnorm import SyncBatchNorm as SyncBN
from torch.optim import AdamW
from mmseg.datasets.transforms import (LoadAnnotations, PackSegInputs,
                                       PhotoMetricDistortion, RandomCrop,
                                       ResizeShortestEdge)
from mmseg.datasets.transforms.loading import LoadSingleRSImageFromFile



# EncoderDecoder
from mmseg.models.segmentors.encoder_decoder_hsm import EncoderDecoder
# SegDataPreProcessor
from mmseg.models.data_preprocessor import SegDataPreProcessor
# Backbone
from mmseg.models.backbones.deit3 import DeiT3
from mmpretrain.models.backbones.convnext import ConvNeXt
# Neck
from mmseg.models.necks.multilevel_neck import  MultiLevelNeck
# DecodeHead
from mmseg.models.decode_heads.uper_head_hsm import UPerHead_HSM
# Loss
from mmseg.models.losses.hcc_loss import HCC_LOSS

# Optimizer
from mmseg.engine.optimizers import (LayerDecayOptimizerConstructor,
                                     LearningRateDecayOptimizerConstructor)
# Evaluation
from mmseg.evaluation import IoUMetric
from mmseg.evaluation.metrics.iou_metric_level import IoUMetric_level

with read_base():
    from ..._base_.datasets.GF2_5B_18class_640 import *
    from ..._base_.default_runtime import *
    from ..._base_.schedules.schedule_80k import *

ouput_level = 'L3'  # 输出L3, 验证L3的精度
results_path_merge = True  # 是否合并层级结果


L1_num_classes = 4  # number of L1 Level label   # 5
L2_num_classes = 9  # number of L1 Level label  # 11  5+11+21=37类
L3_num_classes = 18  # number of L1 Level label  # 21

crop_size = (640, 640)
norm_cfg = dict(type=SyncBN, requires_grad=True)

pretrained = 'checkpoints/2-对比实验的权重/deit3/4chan/deit3-base-384px-4chan.pth'
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
    # pretrained=None,
    backbone=dict(
        type=DeiT3,
        arch='base',
        in_channels=4,
        img_size=crop_size[0],
        patch_size=16,
        drop_path_rate=0.15,
        out_type='featmap',
        out_indices=(2, 5, 8, 11), # -1 ?测试一下
        init_cfg=dict(type='Pretrained', checkpoint=pretrained, prefix='backbone.'),
        ),
    neck=dict(
        type=MultiLevelNeck,
        in_channels=[768, 768, 768, 768],
        out_channels=768,
        scales=[4, 2, 1, 0.5]),
    
    decode_head=dict(
        type=UPerHead_HSM,
        ouput_level = ouput_level,
        path_merge = results_path_merge,
        num_classes_level_list = [L1_num_classes, L2_num_classes, L3_num_classes],
        hiera_mode = 'xiaorong5',
        loss_decode=dict(
            type=HCC_LOSS,
            mode = 'HCC',
            num_classes=[L1_num_classes, L2_num_classes, L3_num_classes],
            loss_weight=1.0),
        # type=UPerHead,
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=768,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

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

    # loss_scale='dynamic')
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
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
    checkpoint=dict(type=CheckpointHook, by_epoch=False, interval=2000, max_keep_ckpts=2),
    sampler_seed=dict(type=DistSamplerSeedHook),
    visualization=dict(type=SegVisualizationHook))

val_evaluator = dict(
    type=IoUMetric, iou_metrics=['mIoU', 'mFscore'])  # 'mDice', 'mFscore'
test_evaluator = dict(
    type=IoUMetric_level,
    is_baseline = False,
    test_output_level = ouput_level,
    num_classes_list = [4,9,18],
    iou_metrics=['mIoU', 'mFscore'],
    # format_only=True,
    keep_results=True)