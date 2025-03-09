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
from mmseg.models.backbones.internvit_6b import InternViT6B
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
    from ..._base_.datasets.a_atl_0_paper_5b_GF2_18class_224 import *
    from ..._base_.default_runtime import *
    from ..._base_.schedules.schedule_80k import *

num_classes= 18 

norm_cfg = dict(type=SyncBN, requires_grad=True)
pretrained = 'checkpoints/2-对比实验的权重/piip/InternViT-6B/intern_vit_6b_224px.pth',

crop_size = (640, 640)
data_preprocessor = dict(
    type=SegDataPreProcessor,
    mean =[454.1608733420, 320.6480230485 , 238.9676917808 , 301.4478970428],
    std =[55.4731833972, 51.5171917858, 62.3875607521, 82.6082214602],
    # bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
    test_cfg=dict(size_divisor=32))


model = dict(
    type=EncoderDecoder,
    data_preprocessor=data_preprocessor,
    # pretrained=pretrained,
    backbone=dict(
        type=InternViT6B,
        in_chans=4, 
        img_size=640,
        pretrain_img_size=224,
        patch_size=16,
        pretrain_patch_size=14,  #邪门儿
        depth=48,
        embed_dim=3200,
        num_heads=25,
        mlp_ratio=4,
        qkv_bias=False,
        init_values=0.1,
        with_cp=True,
        use_flash_attn=True,
        qk_normalization=True,
        layerscale_force_fp32=False,
        drop_path_rate=0.4,
        with_fpn=True,    # 金字塔
        out_indices=[11, 23, 35, 47],
        pretrained=pretrained,
        norm_layer_type='RMSNorm',
        output_dtype='float32',
        mlp_type='fused_mlp'
    ),

    decode_head=dict(
        type=UPerHead,
        in_channels=[3200, 3200, 3200, 3200],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=1536,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0)
    ),
  
    # auxiliary_head=dict(
    #     type=FCNHead,
    #     in_channels=3200,
    #     in_index=2,
    #     channels=1536,
    #     num_convs=1,
    #     concat_input=False,
    #     dropout_ratio=0.1,
    #     num_classes=num_classes,
    #     norm_cfg=norm_cfg,
    #     align_corners=False,
    #     loss_decode=dict(
    #         type=CrossEntropyLoss, use_sigmoid=False, loss_weight=0.4)
    # ),

    # test_cfg=dict(mode='slide', crop_size=(640, 640), stride=(196, 196))
    test_cfg=dict(mode='whole')
    # test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341))
)

optimizer = dict(
    type=AdamW,
    lr=4e-5,
    betas=(0.9, 0.999),
    weight_decay=0.05,
)

optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=optimizer,
    constructor=CustomLayerDecayOptimizerConstructor,
    paramwise_cfg=dict(num_layers=48, layer_decay_rate=0.95))

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

