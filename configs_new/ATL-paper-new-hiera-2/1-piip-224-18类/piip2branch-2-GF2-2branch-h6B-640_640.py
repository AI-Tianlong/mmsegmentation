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


norm_cfg = dict(type=SyncBN, requires_grad=True)

# deepspeed = True
deepspeed = False
deepspeed_config = 'configs_zero_deepspeed/adam_zero1_bf16.json'

data_preprocessor = dict(
    type=SegDataPreProcessor,
    mean =[454.1608733420, 320.6480230485 , 238.9676917808 , 301.4478970428],
    std =[55.4731833972, 51.5171917858, 62.3875607521, 82.6082214602],
    # bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
    test_cfg=dict(size_divisor=32))



pretrained = None
model = dict(
    type=EncoderDecoder,
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type=PIIPTwoBranch,
        n_points=4,
        deform_num_heads=16,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        with_cffn=True,
        interact_attn_type='deform',
        interaction_drop_path_rate=0.4,
        interaction_proj=False,
        norm_layer='none',
        # InternViT-6B
        branch1=dict(
            real_size=640, 
            img_size=640,
            pretrain_img_size=224,
            patch_size=16,
            pretrain_patch_size=14,
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
            with_fpn=False,
            drop_path_rate=0.4,
            interaction_indexes=[[0, 3], [4, 7], [8, 11], [12, 15], [16, 19], [20, 23], [24, 27], [28, 31], [32, 35], [36, 39], [40, 43], [44, 47]], #TODO
            pretrained = 'checkpoints/2-对比实验的权重/piip/InternViT-6B/intern_vit_6b_224px.pth',
            norm_layer_type='RMSNorm',
            mlp_type='fused_mlp',
        ),
        # InternViT-Huge
        branch2=dict(
            real_size=640,
            img_size=640,
            pretrain_img_size=224,
            patch_size=16,
            pretrain_patch_size=14,
            depth=32,
            embed_dim=1280,
            num_heads=16,
            mlp_ratio=4,
            qkv_bias=True,
            init_values=1.0,
            with_cp=True,
            use_flash_attn=True,
            qk_normalization=False,
            layerscale_force_fp32=False,
            with_fpn=False,
            drop_path_rate=0.4,
            interaction_indexes=[[0, 1], [2, 3], [4, 5], [6, 7], [8, 10], [11, 13], [14, 16], [17, 19], [20, 22], [23, 25], [26, 28], [29, 31]], #TODO
            pretrained = 'checkpoints/2-对比实验的权重/piip/InternViT-6B/mae_pretrain_vit_huge.pth',
            norm_layer_type='LayerNorm',
            mlp_type='fused_mlp'
        ),
    ),

    decode_head=dict(
        type=UPerHead,
        in_channels=[3200, 3200, 3200, 3200],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=1536,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0)
    ),
  
    auxiliary_head=dict(
        type=FCNHead,
        in_channels=3200,
        in_index=2,
        channels=1536,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type=CrossEntropyLoss, use_sigmoid=False, loss_weight=0.4)
    ),

    test_cfg=dict(mode='slide', crop_size=(224, 224), stride=(196, 196))
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
    paramwise_cfg=dict(num_layers=48, layer_decay_rate=0.95, skip_stride=1.5))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)


# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)
runner = dict(type='IterBasedRunner')
if deepspeed:
    checkpoint_config = dict(deepspeed=deepspeed, by_epoch=False, interval=2000, max_keep_ckpts=1)
else:
    checkpoint_config = dict(by_epoch=False, interval=2000, max_keep_ckpts=1)
evaluation = dict(interval=1000, metric='mIoU', save_best=None)

if deepspeed:
    custom_hooks = [
        dict(
            type='ToBFloat16Hook',
            priority=49),
    ]
