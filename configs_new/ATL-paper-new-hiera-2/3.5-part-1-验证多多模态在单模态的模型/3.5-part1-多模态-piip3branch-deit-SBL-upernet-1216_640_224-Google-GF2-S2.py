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
from mmseg.models.data_preprocessor_atl import ATL_SegDataPreProcessor
from mmseg.models.data_preprocessor_multimodal import ATL_SegDataPreProcessor_MultiModal
# Backbone
from mmseg.models.backbones.mscan import MSCAN
from mmseg.models.backbones.piip_2branch import PIIPTwoBranch
from mmseg.models.backbones.piip_3branch import PIIPThreeBranch
from mmseg.models.backbones.internvit_6b import InternViT6B
from mmseg.models.backbones.piip_3branch_multimodal import PIIPThreeBranch_MultiModal
# Neck
from mmseg.models.necks.multilevel_neck import  MultiLevelNeck
# DecodeHead
from mmseg.models.decode_heads.uper_head import UPerHead
from mmseg.models.decode_heads.fcn_head import FCNHead
from mmseg.models.decode_heads.uper_head_MultiModal import UPerHead_MultiModal
# Loss
from mmseg.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmseg.models.losses.atl_hiera_37_loss_convseg import ATL_Hiera_Loss_convseg
#optimizer
from mmseg.engine.optimizers.piip_layer_decay_optimizer_constructor import CustomLayerDecayOptimizerConstructor

# Evaluation
from mmseg.evaluation import IoUMetric
from mmseg.evaluation.metrics.iou_metric_MultiModal import IoUMetric_MultiModal

with read_base():
    from ..._base_.datasets.a_atl_0_paper_multi_Google_GF2_S2_18class import *
    from ..._base_.default_runtime import *
    from ..._base_.schedules.schedule_80k import *


norm_cfg = dict(type=SyncBN, requires_grad=True)
num_classes = 18

# 这和后面base的模型不一样的话，如果在decode_head里，给这三个数赋值的话，会报非常难定的错误
crop_size = (1216, 640, 224)

pretrained_large_branch1_10chan = 'checkpoints/2-对比实验的权重/piip/deit/10chan/deit_10chan_large_224_21k.pth'
pretrained_base_branch2_4chan   = 'checkpoints/2-对比实验的权重/piip/deit/4chan/deit_4chan_base_224_21k.pth'
pretrained_small_branch3_3chan  = 'checkpoints/2-对比实验的权重/piip/deit/3chan/deit_3chan_small_224_21k.pth'

data_preprocessor = dict(
        type=ATL_SegDataPreProcessor_MultiModal,
        mean = None,
        std = None,
        pad_val=0,
        seg_pad_val=255,
        size=crop_size,
        test_cfg=dict(size=crop_size))  # 已经在dataloader resize了，这里只是paddy


model = dict(
    type=EncoderDecoder,
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type=PIIPThreeBranch_MultiModal,
        n_points=4,
        deform_num_heads=16,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        with_cffn=True,
        interact_attn_type='deform',
        interaction_drop_path_rate=0.4,
        interaction_proj=False,
        norm_layer='none',
        # For S2 10 band ViT-large
        branch1=dict(
            in_chans=10, 
            real_size=crop_size[2],
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
            with_fpn=False,
            interaction_indexes=[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21], [22, 23]],
            pretrained = pretrained_large_branch1_10chan,
            use_flash_attn=True,
            window_attn=[True, True, True, True, True, True,
                        True, True, True, True, True, True,
                        True, True, True, True, True, True,
                        True, True, True, True, True, True,],
            window_size=[28, 28, 28, 28, 28, 28,
                        28, 28, 28, 28, 28, 28,
                        28, 28, 28, 28, 28, 28,
                        28, 28, 28, 28, 28, 28],

        ),
        # For GF2 4 band ViT-base
        branch2=dict(
            in_chans=4, 
            real_size=crop_size[1],
            pretrain_img_size=224,
            patch_size=16,
            pretrain_patch_size=16,
            depth=12,
            embed_dim=768,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            drop_path_rate=0.15,
            init_scale=1.,
            with_fpn=False,
            interaction_indexes=[[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10], [11, 11]],
            pretrained = pretrained_base_branch2_4chan,
            use_flash_attn=True,
            window_attn=[True, True, True,
                        True, True, True,
                        True, True, True,
                        True, True, True,],
            window_size=[28, 28, 28,
                        28, 28, 28,
                        28, 28, 28,
                        28, 28, 28],
        ),
        # For Google 3 band ViT-small
        branch3=dict(
            in_chans=3, 
            real_size=crop_size[0],
            pretrain_img_size=224,
            patch_size=16,
            pretrain_patch_size=16,
            depth=12,
            embed_dim=384,
            num_heads=6,
            mlp_ratio=4,
            qkv_bias=True,
            drop_path_rate=0.05,
            init_scale=1.,
            with_fpn=False,
            interaction_indexes=[[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10], [11, 11]],
            pretrained = pretrained_small_branch3_3chan,
            use_flash_attn=True,
            window_attn=[True, True, True,
                         True, True, True,
                         True, True, True,
                         True, True, True,],
            window_size=[28, 28, 28,
                        28, 28, 28,
                        28, 28, 28,
                        28, 28, 28],
        ),
    ),
    # neck=dict(
    #     type=MultiLevelNeck,
    #     in_channels=[1024, 1024, 1024, 1024],
    #     out_channels=1024,
    #     scales=[4, 2, 1, 0.5]),

    decode_head=dict(
        type=UPerHead_MultiModal,
        in_channels=[1024, 1024, 1024, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=1024,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0)
    ),
  
    # auxiliary_head=dict(
    #     type=FCNHead,
    #     in_channels=1024,
    #     in_index=3,
    #     channels=256,
    #     num_convs=1,
    #     concat_input=False,
    #     dropout_ratio=0.1,
    #     num_classes=num_classes,
    #     norm_cfg=norm_cfg,
    #     align_corners=False,
    #     loss_decode=dict(
    #         type=CrossEntropyLoss, use_sigmoid=False, loss_weight=0.4)
    # ),

    test_cfg=dict(mode='whole')
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
train_cfg = dict(type=IterBasedTrainLoop, max_iters=80000, val_interval=8000)
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
    type=IoUMetric_MultiModal, 
    iou_metrics=['mIoU', 'mFscore'])  # 'mDice', 'mFscore'
test_evaluator = dict(
    type=IoUMetric_MultiModal,
    iou_metrics=['mIoU', 'mFscore'],
    # format_only=True,
    keep_results=True)
