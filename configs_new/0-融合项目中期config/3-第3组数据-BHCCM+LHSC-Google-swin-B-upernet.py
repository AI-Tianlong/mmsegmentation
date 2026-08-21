from mmengine.config import read_base
from mmengine.optim.optimizer import OptimWrapper
from mmengine.optim.scheduler.lr_scheduler import LinearLR, PolyLR
from torch.nn.modules.batchnorm import SyncBatchNorm as SyncBN
from torch.optim import AdamW

# EncoderDecoder
from mmseg.models.segmentors.encoder_decoder_BHCCM import EncoderDecoder_BHCCM
# SegDataPreProcessor
from mmseg.models.data_preprocessor import SegDataPreProcessor
# Backbone
from mmseg.models.backbones.swin import SwinTransformer
# DecodeHead
from mmseg.models.decode_heads.uper_head_BHCCM import UPerHead_BHCCM

# Loss
# from mmseg.models.losses.hcc_loss import HCC_LOSS
from mmseg.models.losses.atl_hsc_loss import HSC_LOSS

# Evaluation
from mmseg.evaluation.metrics.iou_metric_hsm import IoUMetric_HSM

with read_base():
    from .._base_.datasets.Google_5B_18class_896 import *
    from .._base_.default_runtime import *
    from .._base_.schedules.schedule_80k import *



# 第一组数据集配置
data_root = 'data/1-paper-segmentation/2-多领域地物覆盖基础/Google_5B_19类/5-裁切好的图像/2-三组数据集/第三组'

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/val',
            seg_map_path='ann_dir/val'),
        pipeline=test_pipeline))


# base setting 
ouput_level = 'L3'        # 输出L3, 验证L3的精度
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

L1_num_classes = 4   # number of L1 Level label   # 5
L2_num_classes = 9   # number of L1 Level label  # 11  5+11+21=37类
L3_num_classes = 18  # number of L1 Level label  # 21

crop_size = (896, 896)
norm_cfg = dict(type=SyncBN, requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)

# pretrained = 'checkpoints/2-对比实验的权重/swin-224/base/swin_base_win7_224_3chan.pth'
pretrained = None
data_preprocessor = dict(
    type=SegDataPreProcessor,
    mean = [123.675, 116.28, 103.53],
    std = [58.395, 57.12, 57.375],
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

model = dict(
    type=EncoderDecoder_BHCCM,
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type=SwinTransformer,
        in_channels=3,
        pretrain_img_size=224,
        embed_dims=128,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=backbone_norm_cfg,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained),
        ),
    decode_head=dict(
        type=UPerHead_BHCCM,
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
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

optimizer=dict(
    type=AdamW, 
    lr=0.00006, 
    betas=(0.9, 0.999), 
    weight_decay=0.01)

optim_wrapper = dict(
    # type='AmpOptimWrapper',  # mmengine 混合精度江都训练内存
    type=OptimWrapper,
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

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

train_cfg.update(type=IterBasedTrainLoop, max_iters=80000, val_interval=8000) # 4000
default_hooks.update(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(type=CheckpointHook, by_epoch=False, interval=2000, max_keep_ckpts=10),
    sampler_seed=dict(type=DistSamplerSeedHook),
    visualization=dict(type=SegVisualizationHook))

