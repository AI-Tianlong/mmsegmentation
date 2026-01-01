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
from mmseg.models.segmentors.encoder_decoder_BHCCM import EncoderDecoder_BHCCM
# SegDataPreProcessor
from mmseg.models.data_preprocessor import SegDataPreProcessor
# Backbone
from mmseg.models.backbones.resnet import ResNetV1c
# DecodeHead
from mmseg.models.decode_heads.uper_head_BHCCM import UPerHead_BHCCM
from mmseg.models.decode_heads.sep_aspp_head_BHCCM import DepthwiseSeparableASPPHead_BHCCM
# Loss
from mmseg.models.losses.atl_hsc_loss import HSC_LOSS
# Optimizer
from torch.optim.sgd import SGD
from mmseg.engine.optimizers import (LayerDecayOptimizerConstructor,
                                     LearningRateDecayOptimizerConstructor)
# Evaluation
from mmseg.evaluation.metrics.iou_metric_hsm import IoUMetric_HSM

with read_base():
    from ..._base_.datasets.GF2_5B_18class_640 import *
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

crop_size = (640, 640)
norm_cfg = dict(type=SyncBN, requires_grad=True)

pretrained = 'checkpoints/2-对比实验的权重/deeplabv3plus/resnet101_v1c-4chan.pth'
data_preprocessor = dict(
        type=SegDataPreProcessor,
        mean = [412.62603765, 317.66892688, 243.74720123, 292.61469172],
        std = [42.79585263, 45.59081086, 54.94280476, 69.32133677],
        pad_val=0,
        seg_pad_val=255,
        size=crop_size)

model = dict(
    type=EncoderDecoder_BHCCM,
    data_preprocessor=data_preprocessor,
    pretrained=pretrained,
    backbone=dict(
        type=ResNetV1c,
        depth=101,
        in_channels = 4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type=DepthwiseSeparableASPPHead_BHCCM,
        ouput_level = ouput_level,
        results_with_JSPS = results_with_JSPS,
        num_classes_level_list = [L1_num_classes, L2_num_classes, L3_num_classes],
        hiera_mode = 'xiaorong6', # 双向的+多一个原始特征交互。
        loss_decode=dict(
            type=HSC_LOSS,
            mode = 'L_HSC',
            num_classes=[L1_num_classes, L2_num_classes, L3_num_classes],
            loss_weight=1.0),
        
        # type=UPerHead,
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

optimizer=dict(
        type=AdamW, 
        lr=0.0001, 
        betas=(0.9, 0.999), 
        weight_decay=0.05)

optimizer = dict(type=SGD, lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type=OptimWrapper, optimizer=optimizer, clip_grad=None)

param_scheduler = [
    dict(
        type=PolyLR,
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=80000,
        by_epoch=False)
]

train_cfg.update(type=IterBasedTrainLoop, max_iters=80000, val_interval=8000) # 4000
default_hooks.update(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(type=CheckpointHook, by_epoch=False, interval=2000, max_keep_ckpts=4),
    sampler_seed=dict(type=DistSamplerSeedHook),
    visualization=dict(type=SegVisualizationHook))

