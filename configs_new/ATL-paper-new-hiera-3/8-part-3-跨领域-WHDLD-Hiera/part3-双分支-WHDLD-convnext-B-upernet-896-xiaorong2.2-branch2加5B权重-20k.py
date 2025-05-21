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
# from mmpretrain.models.backbones.convnext import ConvNeXt
from mmseg.models.backbones.convnext import ConvNeXt
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

# 2branch 相关的
from mmseg.models.segmentors.twobranch_encoder_decoder_mode2 import TwoBranch_EncoderDecoder_mode2
from mmseg.models.backbones.atl_twobranch_backbone_mode2 import TwoBranch_backbone_mode2
from mmseg.models.decode_heads.atl_twobranch_head_mode2_isaid import TwoBranch_decode_head_mode2_iSAID  
from mmseg.models.decode_heads.uper_head_hiera_with_gate import UPerHeadWithGate
from mmseg.models.decode_heads.uper_head_hiera_with_gate_whdld import UPerHeadWithGate_WHDLD



with read_base():
    from ..._base_.datasets.part3_isaid_tank  import *
    from ..._base_.default_runtime import *
    from ..._base_.schedules.schedule_20k import *

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
new_task_classes = 6

crop_size = (896, 896)
norm_cfg = dict(type=SyncBN, requires_grad=True)

imagenet_pretrained = 'checkpoints/2-对比实验的权重/convnext/base/convnext-base-3chan.pth'
land_use_checkpoint = 'checkpoints/part3-双分支/Google-18类-Hiera/Google-5B-convnext-B-upernet-Hiera-miou56.57-68.65.pth'

data_preprocessor = dict(
    type=SegDataPreProcessor,
    mean = [58.842848228996296, 55.30874234401325, 56.17873877879168],  
    std = [26.394018607263266,  21.885492331506306, 21.72762947650889],
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

model = dict(
    type=TwoBranch_EncoderDecoder_mode2,
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type=TwoBranch_backbone_mode2,
        n_points=4,
        deform_num_heads=16,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        with_cffn=True,
        interact_attn_type='deform',  # 'deform' or 'normal'
        interaction_drop_path_rate=0.4,
        with_simple_fpn=False,
        out_interaction_indexes=[0, 1, 10], # 这个是?
        
        branch1_backbone=dict(  # land use
            type=ConvNeXt,
            init_cfg=dict(type='Pretrained', checkpoint=land_use_checkpoint, prefix='backbone.'),   
            in_channels=3,
            arch='base',
            out_indices=[0, 1, 2, 3],
            drop_path_rate=0.4,
            layer_scale_init_value=1.0,
            gap_before_final_norm=False,
            
            img_real_size = crop_size[0],
            interaction_indexes=[
                [0, 2], 
                [3, 6], 
                [7, 10], [11, 13], [14, 16], [17, 19], [20, 22], [23, 25], [26, 28], [29, 31], [32, 34],
                [35, 38],
            ],
            ),
        branch2_backbone=dict( # new_task
            type=ConvNeXt,
            init_cfg=dict(type='Pretrained', checkpoint=land_use_checkpoint, prefix='backbone.'),  # 这里可以消融一下
            in_channels=3,
            arch='base',
            out_indices=[0, 1, 2, 3],
            drop_path_rate=0.4,
            layer_scale_init_value=1.0,
            gap_before_final_norm=False,
            
            img_real_size = crop_size[0],
            interaction_indexes=[  # 0-38 共39个 
                [0, 2],   # 0 1 2      block  block block
                [3, 6],   # 3 4 5 6    下采样 block
                [7, 10], [11, 13], [14, 16], [17, 19], [20, 22], [23, 25], [26, 28], [29, 31], [32, 34],
                [35, 38],
            ],
            ), 
    ),
    decode_head=dict(
        type=TwoBranch_decode_head_mode2_iSAID,
        mode = 'xiaorong2',
        branch1_decode_head=dict(  # land use
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
            norm_cfg=dict(type=SyncBN, requires_grad=False),
            align_corners=False),

        branch2_decode_head=dict( # new_task
            type=UPerHeadWithGate_WHDLD,
            dataset = 'WHDLD',
            mode='xiaorong2-2-2', 
            land_use_level_num = 2,
            # init_cfg=dict(type='Pretrained', checkpoint=land_use_checkpoint, prefix='decode_head.'),  # 这里也可以消融一下
            in_channels=[128, 256, 512, 1024],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=768,
            dropout_ratio=0.1,
            num_classes=new_task_classes,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0)),
        ),
            
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

optimizer=dict(
        type=AdamW, 
        lr=0.0001, 
        betas=(0.9, 0.999), 
        weight_decay=0.05)
        
# optimizer = dict(type='AdamW', 
#                  lr=0.0002, 
#                  betas=(0.9, 0.999),
#                  weight_decay=0.05,
#                  constructor='CustomLayerDecayOptimizerConstructorMMDet',
#                  paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.8, skip_stride=[1, 3])
#                 )

optim_wrapper = dict(
    # type='AmpOptimWrapper',  # mmengine 混合精度江都训练内存
    type=OptimWrapper,
    optimizer=optimizer,
    constructor=LearningRateDecayOptimizerConstructor,
    paramwise_cfg={
        'decay_rate': 0.9,
        'decay_type': 'stage_wise',
        'num_layers': 12,
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
        end=20000,
        eta_min=0.0,
        by_epoch=False,
    )
]

train_cfg.update(type=IterBasedTrainLoop, max_iters=20000, val_interval=4000)
default_hooks.update(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, interval=1, log_metric_by_epoch=False),
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

