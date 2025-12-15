# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class ZGJ_HuHua_Dataset(BaseSegDataset):
    """"""
    METAINFO = dict(
        classes=('背景类','互花米草'),
        palette=[[255, 255, 255 ], [0,255,255]])       

    def __init__(
        self,
        img_suffix='.tif',
        seg_map_suffix='.tif',
        # img_suffix='.png',
        # seg_map_suffix='png',
        reduce_zero_label=False,  # 这里还是要设置为True，因为实际推理出来的结果是 0+24 类，是有reduce_zero_label的
        **kwargs
    ) -> None:  # 所以推理的时候，会加上一个背景类。
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)

