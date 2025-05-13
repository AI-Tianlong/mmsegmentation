from mmcv.transforms.loading import LoadImageFromFile
from mmcv.transforms.processing import (RandomFlip, RandomResize, Resize,
                                        TestTimeAug)
from mmengine.dataset.sampler import DefaultSampler, InfiniteSampler

from mmseg.datasets.atl_0_paper_5b_s2_22class import ATL_S2_5B_Dataset_18class
from mmseg.datasets.transforms.formatting import PackSegInputs
from mmseg.datasets.transforms.loading import (LoadAnnotations,
                                               LoadSingleRSImageFromFile)
from mmseg.datasets.transforms.transforms import (PhotoMetricDistortion,
                                                  RandomCrop)
from mmseg.evaluation import IoUMetric

# dataset settings
dataset_type = ATL_S2_5B_Dataset_18class
data_root = 'data/1-paper-segmentation/2-多领域地物覆盖基础/0-S2-5B-18-512'

# mean = [412.62603765, 317.66892688, 243.74720123, 292.61469172],
# std = [42.79585263, 45.59081086, 54.94280476, 69.32133677],


crop_size = (512, 512)   # 不要随机增强！！！！
train_pipeline = [
    dict(type=LoadSingleRSImageFromFile),
    dict(type=LoadAnnotations),
   
    dict(type=RandomCrop, crop_size=crop_size, cat_max_ratio=0.75),
    dict(
        type=RandomResize,
        scale=crop_size,
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type=RandomCrop, crop_size=crop_size, cat_max_ratio=0.75),
    dict(type=RandomFlip, prob=0.5),
    # dict(type=Resize, scale=crop_size, keep_ratio=True),  # 只是全部resize成640
    # dict(type=PhotoMetricDistortion), # 多通道 不太能用这个
    dict(type=PackSegInputs)
]

val_pipeline = [  #
    dict(type=LoadSingleRSImageFromFile),
    # dict(type=Resize, scale=crop_size, keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type=LoadAnnotations),
    dict(type=PackSegInputs)
]

test_pipeline = [  #
    dict(type=LoadSingleRSImageFromFile),
    # dict(type=Resize, scale=crop_size, keep_ratio=True),
    dict(type=LoadAnnotations),  # 不需要验证，不用添加 Annotations
    # dict(type=Resize, scale=(6800, 7200), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type=PackSegInputs)
]


train_dataloader = dict(
    batch_size=2,
    num_workers=8,  # numworkers 也会影响！
    persistent_workers=True,
    sampler=dict(type=InfiniteSampler, shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/train/',
            seg_map_path='ann_dir/train/'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/val/',
            seg_map_path='ann_dir/val/'),
        pipeline=val_pipeline))
# 想用大图去推理
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            # img_path='/data/AI-Tianlong/openmmlab/mmsegmentation/data/1-paper-segmentation/2-多领域地物覆盖基础/小样本数据集推理/img_dir'),
            img_path='img_dir/val',
            seg_map_path='ann_dir/val'),
        pipeline=test_pipeline))

val_evaluator = dict(
    type=IoUMetric, iou_metrics=['mIoU', 'mFscore'])  # 'mDice', 'mFscore'
test_evaluator = dict(
    type=IoUMetric,
    iou_metrics=['mIoU', 'mFscore'],
    # format_only=True,
    keep_results=True)
