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
from mmseg.models.backbones.piip_3branch_singlemodal import PIIPThreeBranch_SingleModal
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

# Datasets
from mmseg.datasets.atl_0_paper_5b_s2_22class import ATL_S2_5B_Dataset_18class


with read_base():
    from ..._base_.datasets.a_atl_0_paper_multi_Google_GF2_S2_18class import *
    from ..._base_.default_runtime import *
    from ..._base_.schedules.schedule_80k import *


weights = '/data/AI-Tianlong/openmmlab/mmsegmentation/work_dirs/part1-多模态-piip3branch-deit-SBL-upernet-1216_640_224-Google-GF2-S2-miou72.54/iter_80000.pth'

norm_cfg = dict(type=SyncBN, requires_grad=True)
num_classes = 18


# 这和后面base的模型不一样的话，如果在decode_head里，给这三个数赋值的话，会报非常难定的错误
crop_size = [1216,640,224]

pretrained_large_branch1_10chan = 'checkpoints/2-对比实验的权重/piip/deit/10chan/deit_10chan_large_224_21k.pth'
pretrained_base_branch2_4chan   = 'checkpoints/2-对比实验的权重/piip/deit/4chan/deit_4chan_base_224_21k.pth'
pretrained_small_branch3_3chan  = 'checkpoints/2-对比实验的权重/piip/deit/3chan/deit_3chan_small_224_21k.pth'

data_preprocessor = dict(
        type=SegDataPreProcessor,
        mean = [412.62603765, 317.66892688, 243.74720123, 292.61469172],
        std = [42.79585263, 45.59081086, 54.94280476, 69.32133677],
        pad_val=0,
        seg_pad_val=255,
        size=crop_size[1])  # 已经在dataloader resize了，这里只是paddy


model = dict(
    type=EncoderDecoder,
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type=PIIPThreeBranch_SingleModal,
        single_branch = 'branch2',
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
        branch1=None,
        # For GF2 4 band ViT-base
        branch2=dict(
            in_chans=4, 
            real_size=crop_size[1], # pop 掉
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
            # interaction_indexes=[[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10], [11, 11]],
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
        branch3=None,
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

dataset_type = ATL_S2_5B_Dataset_18class
data_root = 'data/1-paper-segmentation/2-多领域地物覆盖基础/0-Google-GF2-S2-地理配准-dataset-base224'
test_pipeline = [  #
    dict(type=LoadSingleRSImageFromFile),
    dict(type=Resize, scale=crop_size[1], keep_ratio=True),
    dict(type=LoadAnnotations),  # 不需要验证，不用添加 Annotations
    # dict(type=Resize, scale=(6800, 7200), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type=PackSegInputs)
]
test_dataloader.update( 
    dict(
        batch_size=1,
        num_workers=4,
        persistent_workers=True,
        sampler=dict(type=DefaultSampler, shuffle=False),
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            data_prefix=dict(
                img_path='img_dir/val/GF2-5B-18-base224',
                seg_map_path='ann_dir/val/GF2-5B-18-base224'),
            pipeline=test_pipeline))
)


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
    type=IoUMetric, 
    iou_metrics=['mIoU', 'mFscore'])  # 'mDice', 'mFscore'
test_evaluator = dict(
    type=IoUMetric,
    iou_metrics=['mIoU', 'mFscore'],
    # format_only=True,
    keep_results=True)
