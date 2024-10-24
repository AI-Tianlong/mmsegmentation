# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class ATLS2FIveBillionDataset24(BaseSegDataset):
    """ISPRS Potsdam dataset.

    In segmentation map annotation for Potsdam dataset, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """
    METAINFO = dict(
        classes=('industrial area', 'paddy field', 'irrigated field',
                 'dry cropland', 'garden land', 'arbor forest', 'shrub forest',
                 'park', 'natural meadow', 'artificial meadow', 'river',
                 'urban residential', 'lake', 'pond', 'fish pond', 'snow',
                 'bareland', 'rural residential', 'stadium', 'square', 'road',
                 'overpass', 'railway station', 'airport'),
        palette=[[200, 0, 0], [0, 200, 0], [150, 250, 0], [150, 200, 150],
                 [200, 0, 200], [150, 0, 250], [150, 150, 250],
                 [200, 150, 200], [250, 200, 0], [200, 200, 0], [0, 0, 200],
                 [250, 0, 150], [0, 150, 200], [0, 200, 250], [150, 200, 250],
                 [250, 250, 250], [200, 200, 200], [200, 150, 150],
                 [250, 200, 150], [150, 150, 0], [250, 150,
                                                  150], [250, 150, 0],
                 [250, 200, 250], [200, 150, 0]])

    def __init__(
        self,
        img_suffix='.tif',
        seg_map_suffix='.tif',
        reduce_zero_label=True,  # 这里还是要设置为True，因为实际推理出来的结果是 0+24 类，是有reduce_zero_label的
        **kwargs
    ) -> None:  # 所以推理的时候，会加上一个背景类。
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)




@DATASETS.register_module()
class ATLS2FIveBillionDataset5(BaseSegDataset):
    """ISPRS Potsdam dataset.

    In segmentation map annotation for Potsdam dataset, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """
    METAINFO = dict(
        classes=("Vegetation", "water","Man-made",   
                    "BareLand", "snow",),
        palette=[[0, 200, 0], [0, 0, 200], [250, 0, 150], 
                 [200, 200, 200], [255, 255, 255]]
                )              
    
    def __init__(
        self,
        img_suffix='.tif',
        seg_map_suffix='.tif',
        reduce_zero_label=True,  # 这里还是要设置为True，因为实际推理出来的结果是 0+24 类，是有reduce_zero_label的
        **kwargs
    ) -> None:  # 所以推理的时候，会加上一个背景类。
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)