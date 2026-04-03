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
from mmseg.models.segmentors.twobranch_encoder_decoder import TwoBranch_EncoderDecoder

# SegDataPreProcessor
from mmseg.models.data_preprocessor import SegDataPreProcessor
# Backbone
# from mmpretrain.models.backbones.convnext import ConvNeXt
from mmseg.models.backbones.convnext import ConvNeXt #mmseg的convnext!自建的！
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
# from mmseg.models.segmentors.twobranch_encoder_decoder_mode2 import TwoBranch_EncoderDecoder_mode2
# from mmseg.models.backbones.atl_twobranch_backbone_mode2 import TwoBranch_backbone_mode2
# from mmseg.models.decode_heads.atl_twobranch_head_mode2 import TwoBranch_decodeee_head_mode2  

# TransLU 相关的
from mmseg.models.losses.atl_hsc_loss import HSC_LOSS
from mmseg.models.losses.atl_hsc_loss_crop10m import HSC_LOSS_crop10m


from mmseg.models.segmentors.encoder_decoder_TransLU_CDKS_CDSA import EncoderDecoder_TransLU_CDKS_CDSA
from mmseg.models.backbones.TransLU_backbone_CDKS import TransLU_backbone_CDKS # 这个两分支正常CDKS交互
from mmseg.models.decode_heads.uper_head import UPerHead                       # 这里只要一个正常的uperhead
# TransLU的双分支deocde_head，包含两个分支的deocde_head, 1分支的decoder需要改动，2分支直接用BHCCM
from mmseg.models.decode_heads.TransLU_decode_head_CDSA import TransLU_decode_head_CDSA # 两分支decode_head有交互

# branch1 的 decode_head 需要稍微做出改变，以适应有mask的参与。
from mmseg.models.decode_heads.uper_head_BHCCM_TransLU_CDSA import UPerHead_BHCCM_TransLU_CDSA
# branch2 的 decode_head 就是 UperHead+BHCCM
from mmseg.models.decode_heads.uper_head_BHCCM import UPerHead_BHCCM


with read_base():
    from ..._base_.datasets.S2_crop10m_18class_512  import *
    from ..._base_.default_runtime import *
    from ..._base_.schedules.schedule_20k import *

test_output_level = 'L3' # 输出L3, 验证L3的精度
results_with_JSPS = True  # 是否合并层级结果

val_evaluator = dict(
    type=IoUMetric, iou_metrics=['mIoU', 'mFscore'])  # 'mDice', 'mFscore'
test_evaluator = dict(
    type=IoUMetric,
    iou_metrics=['mIoU', 'mFscore'],
    # format_only=True,
    keep_results=True)

find_unused_parameters=True

branch1_L1_num_classes = 2  # 其他 植被
branch1_L2_num_classes = 2  # 其他 耕地
branch1_L3_num_classes = 4  # 其他、水稻、大豆、玉米  
crop_num_classes = 4

branch2_L1_num_classes = 4   #  MM-5B L1
branch2_L2_num_classes = 9   #  MM-5B L2
branch2_L3_num_classes = 18  #  MM-5B L3

crop_size = (512, 512)
norm_cfg = dict(type=SyncBN, requires_grad=True)

imagenet_pretrained = 'checkpoints/2-对比实验的权重/convnext/base/convnext-base-10chan.pth'
MM5B_checkpoint = '/data/AI-Tianlong/openmmlab/mmsegmentation/work_dirs/0-20251230-BHCCM/BHCCM-S2/BHCCM+LHSC-S2-convnext-B-upernet-消融6-61.16-70.58/iter_80000.pth'

# load_from = MM5B_checkpoint
data_preprocessor = dict(
    type=SegDataPreProcessor,
    mean =[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    std =[10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000],
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

model = dict(
    type=EncoderDecoder_TransLU_CDKS_CDSA,
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type=TransLU_backbone_CDKS,
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
            init_cfg=dict(type='Pretrained', checkpoint=MM5B_checkpoint, prefix='backbone.'),   
            in_channels=10,
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
            init_cfg=dict(type='Pretrained', checkpoint=MM5B_checkpoint, prefix='backbone.'),  # 这里可以消融一下
            in_channels=10,
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
        type=TransLU_decode_head_CDSA,
        branch1_decode_head=dict(  # NEW TASK
            type=UPerHead_BHCCM_TransLU_CDSA, # 这里的L1标签，植被，应该来源于另一分支的预测，因此是个log_segoits,所以这里用KL散度？还是交叉熵，很难？
            init_cfg=dict(type='Pretrained', checkpoint=MM5B_checkpoint, prefix='decode_head.'),
            results_with_JSPS = True,
            num_classes_level_list = [branch1_L1_num_classes, branch1_L2_num_classes, branch1_L3_num_classes],
            hiera_mode = 'xiaorong6',
            loss_decode=dict(
                type=HSC_LOSS_crop10m,
                mode = 'L_HSC',
                num_classes=[branch1_L1_num_classes, branch1_L2_num_classes, branch1_L3_num_classes],
                loss_weight=1.0),
            in_channels=[128, 256, 512, 1024],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            norm_cfg=dict(type=SyncBN, requires_grad=False),
            align_corners=False),

        branch2_decode_head=dict( # new_task
            type=UPerHead_BHCCM,
            init_cfg=dict(type='Pretrained', checkpoint=MM5B_checkpoint, prefix='decode_head.'),
            results_with_JSPS = results_with_JSPS,
            num_classes_level_list = [branch2_L1_num_classes, branch2_L2_num_classes, branch2_L3_num_classes],
            hiera_mode = 'xiaorong6',
            loss_decode=dict(
                type=HSC_LOSS,
                mode = 'L_HSC',
                num_classes=[branch2_L1_num_classes, branch2_L2_num_classes, branch2_L3_num_classes],
                loss_weight=1.0),
            in_channels=[128, 256, 512, 1024],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            norm_cfg=dict(type=SyncBN, requires_grad=False),
            align_corners=False),
        ),
            
            
    train_cfg=dict(),
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
    logger=dict(type=LoggerHook, interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(type=CheckpointHook, by_epoch=False, interval=2000, max_keep_ckpts=4),
    sampler_seed=dict(type=DistSamplerSeedHook),
    visualization=dict(type=SegVisualizationHook))



