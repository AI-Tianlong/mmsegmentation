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
from torch.nn.modules.activation import GELU
from torch.nn.modules.normalization import GroupNorm as GN


# EncoderDecoder
from mmseg.models.segmentors.encoder_decoder_BHCCM import EncoderDecoder_BHCCM
# SegDataPreProcessor
from mmseg.models.data_preprocessor import SegDataPreProcessor
# Backbone
from mmseg.models.backbones import MSCAN
# DecodeHead
from mmseg.models.decode_heads.uper_head_BHCCM import UPerHead_BHCCM
from mmseg.models.decode_heads.ham_head_BHCCM import LightHamHead_BHCCM

# Loss
# from mmseg.models.losses.hcc_loss import HCC_LOSS
from mmseg.models.losses.atl_hsc_loss import HSC_LOSS
# Optimizer
from mmseg.engine.optimizers import (LayerDecayOptimizerConstructor,
                                     LearningRateDecayOptimizerConstructor)
# Evaluation
from mmseg.evaluation.metrics.iou_metric_hsm import IoUMetric_HSM


with read_base():
    from ..._base_.datasets.S2_5B_18class_512 import *
    from ..._base_.default_runtime import *
    from ..._base_.schedules.schedule_80k import *

# base setting 
ouput_level = 'L3'  # 输出L3, 验证L3的精度
results_with_JSPS = True  # 是否合并层级结果

test_evaluator = dict(
    type=IoUMetric_HSM,
    baseline_or_HSM = 'HSM',  # baseline 会用L3->L2->L1的方式计算, HSM则会按照实际L1 L2 L3去计算,
    test_output_level = ouput_level,  # 配合results_path_merge 使用
    num_classes_list = [4,9,18],
    iou_metrics=['mIoU', 'mFscore'],
    # format_only=True,
    keep_results=True)

val_evaluator = test_evaluator

L1_num_classes = 4  # number of L1 Level label   # 5
L2_num_classes = 9  # number of L1 Level label  # 11  5+11+21=37类
L3_num_classes = 18  # number of L1 Level label  # 21

crop_size = (512, 512)
norm_cfg = dict(type=SyncBN, requires_grad=True)

pretrained = 'checkpoints/2-对比实验的权重/segnext/large/segnext_mscan_l_10chan.pth'    # noqa
ham_norm_cfg = dict(type=GN, num_groups=32, requires_grad=True)
data_preprocessor = dict(
    type=SegDataPreProcessor,
    mean =[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    std =[10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000],
    # bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

model = dict(
    type=EncoderDecoder_BHCCM,
    data_preprocessor=data_preprocessor,
    # pretrained=None,
    backbone=dict(
        type=MSCAN,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained),
        in_channels=10,
        embed_dims=[64, 128, 320, 512],
        mlp_ratios=[8, 8, 4, 4],
        drop_rate=0.0,
        drop_path_rate=0.3,
        depths=[3, 5, 27, 3],
        attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
        attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
        act_cfg=dict(type=GELU),
        norm_cfg=dict(type=SyncBN, requires_grad=True)),

    decode_head=dict(
        type=LightHamHead_BHCCM,
        ouput_level = ouput_level,
        results_with_JSPS = results_with_JSPS,
        num_classes_level_list = [L1_num_classes, L2_num_classes, L3_num_classes],
        hiera_mode = 'xiaorong6', # 双向的
        loss_decode=dict(
            type=HSC_LOSS,
            mode = 'L_HSC',
            num_classes=[L1_num_classes, L2_num_classes, L3_num_classes],
            loss_weight=1.0),
        
        # type=UPerHead,
        in_channels=[128, 320, 512],
        in_index=[1, 2, 3], # 为啥不要第一个？
        channels=1024,
        ham_channels=1024,
        dropout_ratio=0.1,
        norm_cfg=ham_norm_cfg,
        align_corners=False,
        ham_kwargs=dict(
            MD_S=1,
            MD_R=16,
            train_steps=6,
            eval_steps=7,
            inv_t=100,
            rand_init=True)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))


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

train_cfg.update(type=IterBasedTrainLoop, max_iters=80000, val_interval=8000) # 4000
default_hooks.update(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(type=CheckpointHook, by_epoch=False, interval=2000, max_keep_ckpts=4),
    sampler_seed=dict(type=DistSamplerSeedHook),
    visualization=dict(type=SegVisualizationHook))

