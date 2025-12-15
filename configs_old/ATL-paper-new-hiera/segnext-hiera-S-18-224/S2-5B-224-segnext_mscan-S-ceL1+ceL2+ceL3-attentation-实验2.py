
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
from mmseg.models.backbones import MSCAN
# DecodeHead
from mmseg.models.decode_heads.ham_head import LightHamHead
from mmseg.models.decode_heads.atl_hiera_37_ham_head_multi_convseg import ATL_Hiera_LightHamHead_Multi_convseg
from mmseg.models.decode_heads.atl_hiera_37_ham_head_multi_convseg_baseline import  ATL_Hiera_LightHamHead_Multi_convseg_baseline
from mmseg.models.decode_heads.atl_hiera_37_ham_head_multi_convseg_attention import ATL_Hiera_LightHamHead_Multi_convseg_attentation
# Loss
from mmseg.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmseg.models.losses.atl_hiera_37_loss_convseg import ATL_Hiera_Loss_convseg
# Evaluation
from mmseg.evaluation import IoUMetric


with read_base():
    from ..._base_.datasets.a_atl_0_paper_5b_s2_18class_224 import *
    from ..._base_.default_runtime import *
    from ..._base_.schedules.schedule_80k import *


# 非常的一致，连loss和acc_seg都一模一样，去除了种子的影响
randomness=dict(seed=42, deterministic=True)

find_unused_parameters=True

L1_num_classes = 4  # number of L1 Level label   # 5
L2_num_classes = 9  # number of L1 Level label  # 11  5+11+21=37类
L3_num_classes = 18  # number of L1 Level label  # 21

# model settings
checkpoint_file = 'checkpoints/2-对比实验的权重/segnext/small/segnext_mscan_s_10chan.pth'   # noqa
ham_norm_cfg = dict(type=GN, num_groups=32, requires_grad=True)
crop_size = (224, 224)

data_preprocessor = dict(
    type=SegDataPreProcessor,
    mean =None,
    std =None,
    # bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
    test_cfg=dict(size_divisor=32))


model = dict(
    type=ATL_Hiera_EncoderDecoder,
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type=MSCAN,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        in_channels=10,
        embed_dims=[64, 128, 320, 512],
        mlp_ratios=[8, 8, 4, 4],
        drop_rate=0.0,
        drop_path_rate=0.1,
        depths=[2, 2, 4, 2],
        attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
        attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
        act_cfg=dict(type=GELU),
        norm_cfg=dict(type=SyncBN, requires_grad=True)),
    decode_head=dict(
        # type=ATL_Hiera_LightHamHead_Multi_convseg_baseline,
        # # num_classes_level_list=[5,10,19],
        # num_classes=L3_num_classes,
        # loss_decode=dict(
        #     type=ATL_Hiera_Loss_convseg, num_classes=[5,10,19], loss_weight=1.0),
        
        
        # 经过修改的具有层级结构的
        type=ATL_Hiera_LightHamHead_Multi_convseg_attentation,
        num_classes_level_list=[4,9,18],
        loss_decode=dict(
            type=ATL_Hiera_Loss_convseg, num_classes=[4,9,18], loss_weight=1.0),
        
        # ## 原版的-和baseline-完全一致
        # type=LightHamHead,
        # num_classes=L3_num_classes,
        # loss_decode=dict(
        #     type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0),

        in_channels=[128, 320, 512],
        in_index=[1, 2, 3],
        channels=256,
        ham_channels=256,
        dropout_ratio=0.1,      
        norm_cfg=ham_norm_cfg,
        align_corners=False,
        ham_kwargs=dict(
            MD_S=1,
            MD_R=16,
            train_steps=6,
            eval_steps=7,
            inv_t=100,
            rand_init=True)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# # dataset settings
# train_dataloader = dict(batch_size=16)

# optimizer
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(
        type=AdamW, lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
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

train_cfg.update(type=IterBasedTrainLoop, max_iters=80000, val_interval=8000)
default_hooks.update(
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