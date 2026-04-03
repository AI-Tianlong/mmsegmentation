# Copyright (c) OpenMMLab. All rights reserved.
import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class WHDLDDataset_6(BaseSegDataset):
    """ iSAID: A Large-scale Dataset for Instance Segmentation in Aerial Images
    In segmentation map annotation for iSAID dataset, which is included
    in 16 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '_manual1.png'.
    """

    METAINFO = dict(
        classes=('building', 'road', 'pavement', 'vegetation', 'bare soil','water'),


        palette=[[255,0,0], [255,255, 0], [192, 192, 0], [0, 255, 0], [128, 128, 128],
                 [0, 0, 255]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 ignore_index=255,
                 reduce_zero_label=True, # PNG是1-6 需要删掉
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ignore_index=ignore_index,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
        assert fileio.exists(self.data_prefix['img_path'], backend_args=self.backend_args)

