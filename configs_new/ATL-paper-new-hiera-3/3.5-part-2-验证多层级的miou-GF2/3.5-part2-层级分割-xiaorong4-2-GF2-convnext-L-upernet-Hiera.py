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
# SegDataPreProcessor
from mmseg.models.data_preprocessor import SegDataPreProcessor
# Backbone
from mmpretrain.models.backbones.convnext import ConvNeXt
# DecodeHead
from mmseg.models.decode_heads.uper_head import UPerHead
from mmseg.models.decode_heads.uper_head_hiera import UPerHead_Hiera
# Loss
from mmseg.models.losses.atl_hiera_37_loss import ATL_Hiera_Loss
from mmseg.models.losses.atl_hiera_37_loss_convseg import ATL_Hiera_Loss_convseg
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


# 训好的权重：/data/AI-Tianlong/openmmlab/mmsegmentation/work_dirs/0-最终论文里可用的结果/1月30日之后的结果/part2-层级分割-xiaorong4-2-GF2-convnext-L-upernet-Hiera-miiou75.41/iter_80000.pth
test_output_level = 'L1' # 输出L3, 验证L3的精度
# results_merge_hiera = True
results_merge_hiera = False

find_unused_parameters=True
L1_num_classes = 4  # number of L1 Level label   # 5
L2_num_classes = 9  # number of L1 Level label  # 11  5+11+21=37类
L3_num_classes = 18  # number of L1 Level label  # 21

norm_cfg = dict(type=SyncBN, requires_grad=True)
# pretrained  = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-large_3rdparty_in21k_20220301-e6e0ea0a.pth'
pretrained = 'checkpoints/2-对比实验的权重/convnext/large/convnext-large-4chan.pth'
crop_size = (640, 640)
data_preprocessor = dict(
        type=SegDataPreProcessor,
        mean =[454.1608733420, 320.6480230485 , 238.9676917808 , 301.4478970428],
        std =[55.4731833972, 51.5171917858, 62.3875607521, 82.6082214602],
        pad_val=0,
        seg_pad_val=255,
        size=crop_size)

model = dict(
    type=EncoderDecoder,
    data_preprocessor=data_preprocessor,
    # pretrained=None,
    backbone=dict(
        type=ConvNeXt,
        in_channels=4,
        arch='large',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=pretrained, prefix='backbone.')),
    
    decode_head=dict(
        type=UPerHead_Hiera,
        test_output_level=test_output_level, #最终输出的层级
        num_classes_level_list = [L1_num_classes, L2_num_classes, L3_num_classes],
        results_merge_hiera = results_merge_hiera,
        hiera_mode = 'xiaorong4',
        loss_decode=dict(
            type=ATL_Hiera_Loss_convseg,
            num_classes=[L1_num_classes, L2_num_classes, L3_num_classes],
            loss_weight=1.0),
        
         # 原始的
        # type=UPerHead,
        in_channels=[192, 384, 768, 1536],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=1024,
        dropout_ratio=0.1,
        # num_classes=L3_num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        # loss_decode=dict(
        #     type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0)
    
    ),
    # auxiliary_head=dict(
    #     type=FCNHead,
    #     in_channels=768,
    #     in_index=2,
    #     channels=256,
    #     num_convs=1,
    #     concat_input=False,
    #     dropout_ratio=0.1,
    #     num_classes=L3_num_classes,
    #     norm_cfg=norm_cfg,
    #     align_corners=False,
    #     loss_decode=dict(
    #         type=CrossEntropyLoss, use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    # test_cfg=dict(mode='slide', crop_size=crop_size, stride=(128, 128)))
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
        'num_layers': 12
    },
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
    type=IoUMetric_level,
    test_output_level = test_output_level,
    num_classes_list = [4,9,18],
    iou_metrics=['mIoU', 'mFscore'],
    # format_only=True,
    keep_results=True)