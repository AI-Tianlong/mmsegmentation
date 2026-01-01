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
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
# SegDataPreProcessor
from mmseg.models.data_preprocessor import SegDataPreProcessor
# Backbone
from mmseg.models.backbones.swin import SwinTransformer
# DecodeHead
from mmseg.models.decode_heads.uper_head import UPerHead
# Loss
from mmseg.models.losses.cross_entropy_loss import CrossEntropyLoss

# Optimizer
from mmseg.engine.optimizers import (LayerDecayOptimizerConstructor,
                                     LearningRateDecayOptimizerConstructor)
# Evaluation
from mmseg.evaluation import IoUMetric
from mmseg.evaluation.metrics.iou_metric_level import IoUMetric_level

with read_base():
    from ..._base_.datasets.GF2_5B_18class_640 import *
    from ..._base_.default_runtime import *
    # from ..._base_.models.upernet_beit_potsdam import *
    from ..._base_.schedules.schedule_80k import *

test_output_level = 'L3' # 输出L3, 验证L的精度

L3_num_classes = 18
backbone_norm_cfg = dict(type='LN', requires_grad=True)
norm_cfg = dict(type=SyncBN, requires_grad=True)

# pretrained  = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-large_3rdparty_in21k_20220301-e6e0ea0a.pth'
pretrained = 'checkpoints/2-对比实验的权重/swin-224/large/swin_large_win7_224_4chan.pth'

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
        type=SwinTransformer,
        in_channels=4,
        pretrain_img_size=224,
        embed_dims=192,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=backbone_norm_cfg,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained),
        ),
    decode_head=dict(
        type=UPerHead,
        in_channels=[192, 384, 768, 1536],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=L3_num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
    # test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341)))



optimizer=dict(
    type=AdamW, 
    lr=0.00006, 
    betas=(0.9, 0.999), 
    weight_decay=0.01)

optim_wrapper = dict(
    # type='AmpOptimWrapper',  # mmengine 混合精度江都训练内存
    type=OptimWrapper,
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

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

train_cfg.update(type=IterBasedTrainLoop, max_iters=80000, val_interval=8000)
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
    is_baseline = True,
    test_output_level = test_output_level,
    num_classes_list = [4,9,18],
    iou_metrics=['mIoU', 'mFscore'],
    # format_only=True,
    keep_results=True)
