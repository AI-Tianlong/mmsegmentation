from mmengine.config import read_base
from torch.nn.modules.activation import GELU
from torch.nn.modules.batchnorm import SyncBatchNorm as SyncBN
from torch.nn.modules.normalization import GroupNorm as GN


# EncoderDecoder
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
# SegDataPreProcessor
from mmseg.models.data_preprocessor import SegDataPreProcessor
# Backbone
from mmseg.models.backbones import MSCAN
# DecodeHead
from mmseg.models.decode_heads.ham_head import LightHamHead
# Loss
from mmseg.models.losses.cross_entropy_loss import CrossEntropyLoss
# optimizer
from mmengine.optim.optimizer import OptimWrapper
from mmengine.optim.scheduler.lr_scheduler import LinearLR, PolyLR
from torch.optim import AdamW
# Evaluation
from mmseg.evaluation import IoUMetric
from mmseg.evaluation.metrics.iou_metric_level import IoUMetric_level


with read_base():
    from ..._base_.datasets.S2_crop10m_18class_512 import *
    from ..._base_.default_runtime import *
    from ..._base_.schedules.schedule_80k import *

# test_output_level = 'L3' # 输出L3, 验证L的精度

num_classes = 4 #倒是也不太影像，这里该改成19的
# randomness=dict(seed=42, deterministic=True)   # 同时要去改test.py文件
# find_unused_parameters=True

# model settings
checkpoint_file = 'checkpoints/2-对比实验的权重/segnext/base/segnext_mscan_b_10chan.pth'   # noqa
ham_norm_cfg = dict(type=GN, num_groups=32, requires_grad=True)
crop_size = (512, 512)

data_preprocessor = dict(
    type=SegDataPreProcessor,
    mean =[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    std =[10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000],
    # bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

model = dict(
    type=EncoderDecoder,
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type=MSCAN,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        in_channels=10,
        embed_dims=[64, 128, 320, 512],
        mlp_ratios=[8, 8, 4, 4],
        drop_rate=0.0,
        drop_path_rate=0.1,
        depths=[3, 3, 12, 3],
        attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
        attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
        act_cfg=dict(type=GELU),
        norm_cfg=dict(type=SyncBN, requires_grad=True)),
    decode_head=dict(
        type=LightHamHead,
        in_channels=[128, 320, 512],
        in_index=[1, 2, 3], # 为啥不要第一个？
        channels=512,
        ham_channels=512,
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

# # dataset settings
# train_dataloader = dict(batch_size=16)

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

param_scheduler = [
    dict(
        type=LinearLR, start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type=PolyLR,
        power=1.0,
        begin=1500,
        end=20000,
        eta_min=0.0,
        by_epoch=False,
    )
]

# training schedule for 80k
train_cfg = dict(type=IterBasedTrainLoop, max_iters=20000, val_interval=4000)
val_cfg = dict(type=ValLoop)
test_cfg = dict(type=TestLoop)

default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(type=CheckpointHook, by_epoch=False, interval=2000, max_keep_ckpts=4),
    sampler_seed=dict(type=DistSamplerSeedHook),
    visualization=dict(type=SegVisualizationHook))


val_evaluator = dict(
    type=IoUMetric, iou_metrics=['mIoU', 'mFscore'])  # 'mDice', 'mFscore'
test_evaluator = dict(
    type=IoUMetric,
    iou_metrics=['mIoU', 'mFscore'],
    # format_only=True,
    keep_results=True)
