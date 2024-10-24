from mmcv.transforms.loading import LoadImageFromFile
from mmcv.transforms.processing import (RandomFlip, RandomResize, Resize,
                                        TestTimeAug)
from mmengine.dataset.sampler import DefaultSampler, InfiniteSampler

from mmseg.datasets.five_billion_pixels import FiveBillionPixelsDataset
from mmseg.datasets.transforms.formatting import PackSegInputs
from mmseg.datasets.transforms.loading import (LoadAnnotations,
                                               LoadSingleRSImageFromFile)
from mmseg.datasets.transforms.transforms import (PhotoMetricDistortion,
                                                  RandomCrop)
from mmseg.evaluation import IoUMetric

# dataset settings
dataset_type = FiveBillionPixelsDataset
data_root = 'data/five_billion_pixels'
crop_size = (512, 512)
train_pipeline = [
    dict(type=LoadSingleRSImageFromFile),
    dict(type=LoadAnnotations),
    dict(
        type=RandomResize,
        scale=(512, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type=RandomCrop, crop_size=crop_size, cat_max_ratio=0.75),
    dict(type=RandomFlip, prob=0.5),
    # dict(type=PhotoMetricDistortion), # 多通道 不太能用这个
    dict(type=PackSegInputs)
]
test_pipeline = [  #
    dict(type=LoadSingleRSImageFromFile),
    dict(type=Resize, scale=(512, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type=LoadAnnotations),
    dict(type=PackSegInputs)
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type=LoadSingleRSImageFromFile, backend_args=None),
    dict(
        type=TestTimeAug,
        transforms=[[
            dict(type=Resize, scale_factor=r, keep_ratio=True)
            for r in img_ratios
        ],
                    [
                        dict(type=RandomFlip, prob=0., direction='horizontal'),
                        dict(type=RandomFlip, prob=1., direction='horizontal')
                    ], [dict(type=LoadAnnotations)],
                    [dict(type=PackSegInputs)]])
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type=InfiniteSampler, shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_8bit_NirRGB/train', seg_map_path='ann_dir/train'),
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
            img_path='img_8bit_NirRGB/val', seg_map_path='ann_dir/val'),
        pipeline=test_pipeline))
# 想用大图去推理
# test_dataloader =  dict(
#     batch_size=1,
#     num_workers=4,
#     persistent_workers=True,
#     sampler=dict(type=DefaultSampler, shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root='',
#         data_prefix=dict(img_path='/opt/AI-Tianlong/Datasets/Five-Billion-Pixels/big_inference', seg_map_path='/opt/AI-Tianlong/Datasets/Five-Billion-Pixels/big_inference'),
#         pipeline=test_pipeline))

test_dataloader = val_dataloader
val_evaluator = dict(
    type=IoUMetric, iou_metrics=['mIoU', 'mFscore'])  # 'mDice', 'mFscore'
test_evaluator = val_evaluator