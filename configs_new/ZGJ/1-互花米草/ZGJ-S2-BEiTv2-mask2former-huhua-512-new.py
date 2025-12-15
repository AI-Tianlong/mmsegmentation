# Copyright (c) Shanghai AI Lab. All rights reserved.
from mmcv.transforms import (LoadImageFromFile, RandomChoice,
                             RandomChoiceResize, RandomFlip)
from mmengine.config import read_base
from mmengine.optim.optimizer import OptimWrapper
from mmengine.optim.scheduler.lr_scheduler import LinearLR, PolyLR
from torch.optim import AdamW

from mmseg.datasets.transforms import (LoadAnnotations, PackSegInputs,
                                       PhotoMetricDistortion, RandomCrop,
                                       ResizeShortestEdge)
from mmseg.datasets.transforms.loading import LoadSingleRSImageFromFile
from mmseg.engine.optimizers import LayerDecayOptimizerConstructor
from mmseg.models.backbones import BEiTAdapter
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder

with read_base():
    from ..._base_.datasets.ZGJ_S2_huhua_512 import *
    from ..._base_.default_runtime import *
    from ..._base_.models.mask2former_beit_potsdam import *
    from ..._base_.schedules.schedule_80k import *

# 训练好的权重：/data/AI-Tianlong/ZGJ/mmsegmentation/work_dirs/ZGJ-S2-BEiTv2-mask2former-huhua-512-new/iter_80000.pth

# # bash tools/dist_test.sh /data/AI-Tianlong/ZGJ/mmsegmentation/configs_new/ZGJ/ZGJ-S2-BEiTv2-mask2former-huhua-512-new.py /data/AI-Tianlong/ZGJ/mmsegmentation/work_dirs/ZGJ-S2-BEiTv2-mask2former-huhua-512-new/iter_80000.pth 5 --out /data/AI-Tianlong/ZGJ/Datasets/0-S2-IMG-2024年/X-推理结果/互花米 
# 草-new-new/mask

# load_from = '/data/AI-Tianlong/ZGJ/mmsegmentation/work_dirs/ZGJ-S2-BEiTv2-mask2former-huhua-512/iter_80000.pth'
# load_from = '/data/AI-Tianlong/ZGJ/checkpoints/5B-pretrained/BEiT/5-s2-5billion-24类地物-用5的接着训-黑龙江省tain和val都训-iter80k-mIoU82.69.pth'
num_classes = 2  # loss 要用，也要加 # 加上背景是25类

# 想用大图去推理
test_pipeline = [  #
    dict(type=LoadSingleRSImageFromFile),
    # dict(type=LoadAnnotations),
    dict(type=PackSegInputs)
]
test_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=None,
        data_prefix=dict(
            img_path=
            # '/data/AI-Tianlong/ZGJ/Datasets/0-S2-IMG-2018年/IMG',   # 2018
            # '/data/AI-Tianlong/ZGJ/Datasets/0-S2-IMG-2018年/IMG', # 2019
            # '/data/AI-Tianlong/ZGJ/Datasets/0-S2-IMG-2024年/4-用来多GPU推理的5120图像', # 2024
            # '/data/AI-Tianlong/ZGJ/Datasets/1-S2-IMG-2017-2025年'
            # '/data/AI-Tianlong/ZGJ/Datasets/S2-2018-2019训练用图/0-裁切好的图像/S2-2018-2019互花米草-512/img_dir/train',
            # seg_map_path='/data/AI-Tianlong/ZGJ/Datasets/S2-2018-2019训练用图/0-裁切好的图像/S2-2018-2019互花米草-512/ann_dir/train'
            # '/data/AI-Tianlong/ZGJ/0-互花米草识别结果-new/1-2017至2025-黄河三角洲区域时序识别结果/1-10波段哨兵二号影像uint16'
            '/data/AI-Tianlong/ZGJ/WYQ/data/互花米草带推理img'
        ),
        pipeline=test_pipeline))


val_evaluator = dict(
    type=IoUMetric, iou_metrics=['mIoU', 'mFscore'])  # 'mDice', 'mFscore'
test_evaluator = dict(
    type=IoUMetric,
    iou_metrics=['mIoU', 'mFscore'],
    format_only=True,
    keep_results=True)

crop_size = (512, 512)
# pretrained = None
# pretrained = '/opt/AI-Tianlong/checkpoints/atl_s2_checkpoint/10_channel_beitv2_large_patch16_224_pt1k_ft21k_BGR.pth'
pretrained = None
data_preprocessor.update(
    dict(
        type=SegDataPreProcessor,
        mean =[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        std =[10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000],
        # mean=None,  # 10个波段的均值
        # std=None,
        pad_val=0,
        seg_pad_val=255,
        size=crop_size))

model.update(
    dict(
        type=EncoderDecoder,
        pretrained=pretrained,
        data_preprocessor=data_preprocessor,
        backbone=dict(
            type=BEiTAdapter,
            img_size=512,
            patch_size=16,
            embed_dim=1024,
            in_channels=10,  # 4个波段
            depth=24,
            num_heads=16,
            mlp_ratio=4,
            qkv_bias=True,
            use_abs_pos_emb=False,
            use_rel_pos_bias=True,
            init_values=1e-6,
            drop_path_rate=0.3,
            conv_inplane=64,
            n_points=4,
            deform_num_heads=16,
            cffn_ratio=0.25,
            deform_ratio=0.5,
            with_cp=False,  # set with_cp=True to save memory
            interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
        ),  #backbone 完全一样
        decode_head=dict(
            in_channels=[1024, 1024, 1024, 1024],
            feat_channels=256,
            out_channels=256,
            num_queries=100,
            num_classes=num_classes,
            loss_cls=dict(
                type=CrossEntropyLoss,
                use_sigmoid=False,
                loss_weight=2.0,
                reduction='mean',
                class_weight=[1.0] * num_classes + [0.1]),
        ),
        test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341))))

# dataset config
train_pipeline = [
    dict(type=LoadSingleRSImageFromFile),
    dict(type=LoadAnnotations),
    dict(
        type=RandomChoiceResize,
        scales=[int(x * 0.1 * 512) for x in range(5, 21)],
        resize_type=ResizeShortestEdge,
        max_size=2048),
    dict(type=RandomCrop, crop_size=crop_size, cat_max_ratio=0.75),
    dict(type=RandomFlip, prob=0.5),
    # dict(type=PhotoMetricDistortion),
    dict(type=PackSegInputs)
]
train_dataloader.update(dataset=dict(pipeline=train_pipeline))  # potsdam的变量

# optimizer
optimizer = dict(
    type=AdamW,
    lr=2e-5,
    betas=(0.9, 0.999),
    weight_decay=0.05,
)
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=optimizer,
    constructor=LayerDecayOptimizerConstructor,
    paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.9))

# learning policy

param_scheduler = [
    dict(type=LinearLR, start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type=PolyLR,
        power=1.0,
        begin=1500,
        end=80000,
        eta_min=0.0,
        by_epoch=False,
    )
]


train_cfg.update(type=IterBasedTrainLoop, max_iters=80000, val_interval=500)
default_hooks.update(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(type=CheckpointHook, by_epoch=False, interval=500, max_keep_ckpts=40),
    sampler_seed=dict(type=DistSamplerSeedHook),
    visualization=dict(type=SegVisualizationHook))
