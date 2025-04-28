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
from mmseg.models.segmentors.atl_hiera_37_encoder_decoder import ATL_Hiera_EncoderDecoder
from mmseg.models.segmentors.twobranch_encoder_decoder import TwoBranch_EncoderDecoder

# SegDataPreProcessor
from mmseg.models.data_preprocessor import SegDataPreProcessor
# Backbone
from mmseg.models.backbones.atl_twobranch_backbone import TwoBranch_backbone
from mmpretrain.models.backbones.convnext import ConvNeXt
# DecodeHead
from mmseg.models.decode_heads.uper_head import UPerHead
from mmseg.models.decode_heads.uper_head_hiera import UPerHead_Hiera
from mmseg.models.decode_heads.uper_head_hiera_2branch import UPerHead_Hiera_2branch
# Loss
from mmseg.models.losses.atl_hiera_37_loss import ATL_Hiera_Loss
from mmseg.models.losses.atl_hiera_37_loss_convseg import ATL_Hiera_Loss_convseg
from mmseg.models.losses.cross_entropy_loss import CrossEntropyLoss

# Optimizer
from mmseg.engine.optimizers.layer_decay_optimizer_constructor_atl import LearningRateDecayOptimizerConstructor
# Evaluation
from mmseg.evaluation import IoUMetric

with read_base():
    from ..._base_.datasets.S2_crop10m_18class_512  import *
    from ..._base_.default_runtime import *
    # from ..._base_.models.upernet_beit_potsdam import *
    from ..._base_.schedules.schedule_80k import *

# 训好的权重:/data/AI-Tianlong/openmmlab/mmsegmentation/work_dirs/0-最终论文里可用的结果/1月30日之后的结果/part2-层级分割-xiaorong4-1-S2-deit-L-upernet-Hiera-miou52.52/iter_80000.pth
test_output_level = 'L1' # 输出L3, 验证L3的精度
results_merge_hiera = False

find_unused_parameters=True
branch1_L1_num_classes = 4  # number of L1 Level label   # 5
branch1_L2_num_classes = 9  # number of L1 Level label  # 11  5+11+21=37类
branch1_L3_num_classes = 18  # number of L1 Level label  # 21


branch2_L1_num_classes = 2  # number of L1 Level label   # 5
branch2_L2_num_classes = 2  # number of L1 Level label  # 11  5+11+21=37类
branch2_L3_num_classes = 4  # 水稻、大豆、玉米
crop_num_classes = 4

crop_size = (512, 512)
norm_cfg = dict(type=SyncBN, requires_grad=True)

imagenet_pretrained = 'checkpoints/2-对比实验的权重/convnext/base/convnext-base-10chan.pth'
land_use_checkpoint = 'checkpoints/part3-双分支/S2-18类-Hiera/part2-层级分割-消融6-S2-convnext-B-upernet-Hiera-miou58.47-67.60.pth'

data_preprocessor = dict(
    type=SegDataPreProcessor,
    mean =[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    std =[10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000],
    # bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

# load_from = '/data/AI-Tianlong/openmmlab/mmsegmentation/checkpoints/part3-双分支/S2-18类-Hiera/part3-层级分割-消融6-S2-convnext-B-upernet-Hiera-2branche'

model = dict(
    type=TwoBranch_EncoderDecoder,
    data_preprocessor=data_preprocessor,
        branch1_backbone_land_use=dict(
            type=ConvNeXt,
            init_cfg=dict(type='Pretrained', checkpoint=land_use_checkpoint, prefix='backbone.'),
            in_channels=10,
            arch='base',
            out_indices=[0, 1, 2, 3],
            drop_path_rate=0.4,
            layer_scale_init_value=1.0,
            gap_before_final_norm=False,
            frozen_stages=4), # 0不冻结任何stage，共四个stage, self.num_stages = len(self.depths)
        
        branch2_backbone_new_task=dict(
            type=ConvNeXt,
            init_cfg=dict(type='Pretrained', checkpoint=imagenet_pretrained, prefix='backbone.'),
            in_channels=10,
            arch='base',
            out_indices=[0, 1, 2, 3],
            drop_path_rate=0.4,
            layer_scale_init_value=1.0,
            gap_before_final_norm=False,
            frozen_stages=0),

        branch1_decode_head_land_use=dict(
            type=UPerHead_Hiera_2branch,
            init_cfg=dict(type='Pretrained', checkpoint=land_use_checkpoint, prefix='decode_head.'),
            num_classes_level_list = [branch1_L1_num_classes, branch1_L2_num_classes, branch1_L3_num_classes],
            results_merge_hiera = False,
            hiera_mode = 'xiaorong6',
            loss_decode=dict(
                type=ATL_Hiera_Loss_convseg,
                num_classes=[branch1_L1_num_classes, branch1_L2_num_classes, branch1_L3_num_classes],
                loss_weight=1.0),
            in_channels=[128, 256, 512, 1024],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=768,
            dropout_ratio=0.1,
            norm_cfg=norm_cfg,
            align_corners=False),

        branch2_decode_head_new_task=dict(
            type=UPerHead_Hiera,
            num_classes_level_list = [branch2_L1_num_classes, branch2_L2_num_classes, branch2_L3_num_classes],
            results_merge_hiera = True,
            hiera_mode = 'xiaorong6',
            loss_decode=dict(
                type=ATL_Hiera_Loss_convseg,
                num_classes=[branch2_L1_num_classes, branch2_L2_num_classes, branch2_L3_num_classes],
                loss_weight=1.0),
            in_channels=[128, 256, 512, 1024],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=768,
            dropout_ratio=0.1,
            norm_cfg=norm_cfg,
            align_corners=False),
            
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
optimizer=dict(
        type=AdamW, 
        lr=0.0001, 
        betas=(0.9, 0.999), 
        weight_decay=0.05)

optim_wrapper = dict(
    # type='AmpOptimWrapper',  # mmengine 混合精度江都训练内存
    type=OptimWrapper,
    optimizer=optimizer,
    constructor=LearningRateDecayOptimizerConstructor,
    paramwise_cfg={
        'decay_rate': 0.9,
        'decay_type': 'stage_wise',
        'num_layers': 12,

        'branch1_backbone_land_use': dict(lr_mult=0.0),
        'branch1_decode_head_land_use': dict(lr_mult=0.0),
        }
    )
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
    type=IoUMetric,
    iou_metrics=['mIoU', 'mFscore'],
    # format_only=True,
    keep_results=True)

