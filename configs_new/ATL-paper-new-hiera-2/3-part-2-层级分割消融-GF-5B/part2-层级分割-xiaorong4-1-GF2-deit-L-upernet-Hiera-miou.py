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
from mmseg.models.backbones.piip_3branch import PIIPThreeBranch
from mmseg.models.backbones.internvit_6b import InternViT6B
from mmseg.models.backbones.deit import vit_models
# Neck
from mmseg.models.necks.multilevel_neck import  MultiLevelNeck
# DecodeHead
from mmseg.models.decode_heads.uper_head import UPerHead
from mmseg.models.decode_heads.uper_head_hiera import UPerHead_Hiera
from mmseg.models.decode_heads.fcn_head import FCNHead
# Loss
from mmseg.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmseg.models.losses.atl_hiera_37_loss import ATL_Hiera_Loss
from mmseg.models.losses.atl_hiera_37_loss_convseg import ATL_Hiera_Loss_convseg
#optimizer
from mmseg.engine.optimizers.piip_layer_decay_optimizer_constructor import CustomLayerDecayOptimizerConstructor

# Evaluation
from mmseg.evaluation import IoUMetric


with read_base():
    from ..._base_.datasets.a_atl_0_paper_5b_GF2_18class_640 import *
    from ..._base_.default_runtime import *
    from ..._base_.schedules.schedule_80k import *

find_unused_parameters = True
norm_cfg = dict(type=SyncBN, requires_grad=True)

L1_num_classes = 4  # number of L1 Level label   # 5
L2_num_classes = 9  # number of L1 Level label  # 11  5+11+21=37类
L3_num_classes = 18  # number of L1 Level label  # 21

# deepspeed = True
deepspeed = False
deepspeed_config = 'configs_zero_deepspeed/adam_zero1_bf16.json'

crop_size = (640, 640)
data_preprocessor = dict(
    type=SegDataPreProcessor,
    # mean =[454.1608733420, 320.6480230485 , 238.9676917808 , 301.4478970428],
    # std =[55.4731833972, 51.5171917858, 62.3875607521, 82.6082214602],
    # bgr_to_rgb=True,
    mean = [412.62603765, 317.66892688, 243.74720123, 292.61469172],
    std = [42.79585263, 45.59081086, 54.94280476, 69.32133677],
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
    test_cfg=dict(size_divisor=32))

# pretrained='checkpoints/2-对比实验的权重/piip/deit/4chan/deit_4chan_base_224_21k.pth'
pretrained = 'checkpoints/2-对比实验的权重/piip/deit/4chan/deit_4chan_large_224_21k.pth'

model = dict(
    type=EncoderDecoder,
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type=vit_models,
        in_chans=4, # 3/4
        img_size=640, # 640 
        pretrain_img_size=224,
        patch_size=16,
        pretrain_patch_size=16,
        depth=24,
        embed_dim=1024,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.4,
        init_scale=1.,
        with_fpn=True,
        # interaction_indexes=[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21], [22, 23]],
        pretrained = pretrained,
        use_flash_attn=True,    # 用上这个后，显著降低了计算量啊！！！！
        window_attn=[True, True, True, True, True, True,
                     True, True, True, True, True, True,
                     True, True, True, True, True, True,
                     True, True, True, True, True, True,],
        window_size=[28, 28, 28, 28, 28, 28,
                     28, 28, 28, 28, 28, 28,
                     28, 28, 28, 28, 28, 28,
                     28, 28, 28, 28, 28, 28],
        ),
    # neck=dict(
    #     type=MultiLevelNeck,
    #     in_channels=[1024, 1024, 1024, 1024],
    #     out_channels=1024,
    #     scales=[4, 2, 1, 0.5]),

    decode_head=dict(
        # 改动的
        type=UPerHead_Hiera,
        num_classes_level_list = [L1_num_classes, L2_num_classes, L3_num_classes],
        results_merge_hiera = True,
        hiera_mode = 'xiaorong4',
        loss_decode=dict(
            type=ATL_Hiera_Loss_convseg,
            num_classes=[L1_num_classes, L2_num_classes, L3_num_classes],
            loss_weight=1.0),

        # 原始的
        # type=UPerHead,
        in_channels=[1024, 1024, 1024, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=1024,
        dropout_ratio=0.1,
        # num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        # loss_decode=dict(
        #     type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0)
    ),
  
    # auxiliary_head=dict(
    #     type=FCNHead,
    #     in_channels=768,
    #     in_index=3,
    #     channels=256,
    #     num_convs=1,
    #     concat_input=False,
    #     dropout_ratio=0.1,
    #     num_classes=num_classes,
    #     norm_cfg=norm_cfg,
    #     align_corners=False,
    #     loss_decode=dict(
    #         type=CrossEntropyLoss, use_sigmoid=False, loss_weight=0.4)),

    # test_cfg=dict(mode='whole')
    test_cfg=dict(mode='slide', crop_size=(640, 640), stride=(384, 384))
)

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

if deepspeed:
    checkpoint_config = dict(deepspeed=deepspeed, by_epoch=False, interval=2000, max_keep_ckpts=1)
else:
    checkpoint_config = dict(by_epoch=False, interval=2000, max_keep_ckpts=1)

if deepspeed:
    custom_hooks = [
        dict(
            type='ToBFloat16Hook',
            priority=49),
    ]
