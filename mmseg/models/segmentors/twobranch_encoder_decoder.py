# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import List, Optional

import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import print_log
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .base import BaseSegmentor


@MODELS.register_module()
class TwoBranch_EncoderDecoder(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.

    1. The ``loss`` method is used to calculate the loss of model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2) Call the decode head loss function to forward decode head model and
    calculate losses.

    .. code:: text

     loss(): extract_feat() -> _decode_head_forward_train() -> _auxiliary_head_forward_train (optional)
     _decode_head_forward_train(): decode_head.loss()
     _auxiliary_head_forward_train(): auxiliary_head.loss (optional)

    2. The ``predict`` method is used to predict segmentation results,
    which includes two steps: (1) Run inference function to obtain the list of
    seg_logits (2) Call post-processing function to obtain list of
    ``SegDataSample`` including ``pred_sem_seg`` and ``seg_logits``.

    .. code:: text

     predict(): inference() -> postprocess_result()
     infercen(): whole_inference()/slide_inference()
     whole_inference()/slide_inference(): encoder_decoder()
     encoder_decoder(): extract_feat() -> decode_head.predict()

    3. The ``_forward`` method is used to output the tensor by running the model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2)Call the decode head forward function to forward decode head model.

    .. code:: text

     _forward(): extract_feat() -> _decode_head.forward()

    Args:

        backbone (ConfigType): The config for the backnone of segmentor.
        decode_head (ConfigType): The config for the decode head of segmentor.
        neck (OptConfigType): The config for the neck of segmentor.
            Defaults to None.
        auxiliary_head (OptConfigType): The config for the auxiliary head of
            segmentor. Defaults to None.
        train_cfg (OptConfigType): The config for training. Defaults to None.
        test_cfg (OptConfigType): The config for testing. Defaults to None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
        pretrained (str, optional): The path for pretrained model.
            Defaults to None.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.
    """  # noqa: E501

    def __init__(self,
                
                 # backbone
                 branch1_backbone_land_use: ConfigType,
                 branch2_backbone_new_task: ConfigType,
                 # decode_head
                 branch1_decode_head_land_use: ConfigType,
                 branch2_decode_head_new_task: ConfigType,

                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
      
        # Build multi backbone 
        self.branch1_backbone_land_use = MODELS.build(branch1_backbone_land_use)
        self.branch2_backbone_new_task = MODELS.build(branch2_backbone_new_task)

        # Build multi decode_head 
        self.branch1_decode_head_land_use = MODELS.build(branch1_decode_head_land_use)
        self.branch2_decode_head_new_task = MODELS.build(branch2_decode_head_new_task)

        self.branch1_land_use_out_channels = self.branch1_decode_head_land_use.out_channels
        self.branch2_new_task_out_channels = self.branch2_decode_head_new_task.out_channels

        self.align_corners = self.branch1_decode_head_land_use.align_corners
        
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg



    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        # import pdb;pdb.set_trace()
        x_land_use = self.branch1_backbone_land_use(inputs)
        x_new_task = self.branch2_backbone_new_task(inputs)
        
        return x_land_use, x_new_task


    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x_land_use, x_new_task = self.extract_feat(inputs)
        land_use_seg_logits = self.branch1_decode_head_land_use.predict(x_land_use, batch_img_metas, self.test_cfg)
        new_task_seg_logits = self.branch2_decode_head_land_use.predict(x_new_task, batch_img_metas, self.test_cfg)

        return land_use_seg_logits, new_task_seg_logits

    def _decode_head_forward_train(self, inputs: List[Tensor],
                                   data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        

        losses = dict()
        loss_decode = self.decode_head.loss(inputs, data_samples, self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        import pdb;pdb.set_trace()                          # [2, 128, 128, 128] [2, 256, 64, 64] [2, 512, 32, 32] [2, 1024, 16, 16]
        x_land_use, x_new_task = self.extract_feat(inputs)  # [2,10,512,512] --> x_land_use(tuple[4级]), x_new_task(tuple[4级])
        x_land_use_list, _ = self.branch1_decode_head_land_use.forward(x_land_use)
        x_new_task_list, _ = self.branch2_decode_head_new_task.forward(x_new_task) #[2, 128, 128, 128] [2, 256, 64, 64] [2, 512, 32, 32] [2, 1024, 16, 16]
        # x_land_use_list = [x_land_use_list[0], x_land_use_list[1], x_land_use_list[2], x_land_use_list[3]]

        # 保存 inputs[0]
        import numpy as np
        from osgeo import gdal
        from ATL_Tools.ATL_gdal import save_array_to_tif
        from PIL import Image
        import os
        import cv2

        inputs_0 = inputs[0]
        inputs_0_np = inputs_0.detach().cpu().numpy().astype(np.float32)
        tif_save_path = '/data/AI-Tianlong/openmmlab/mmsegmentation/configs_new/ATL-paper-new-hiera-3/7-part-3-跨领域-crop10m-Hiera/land_use_分支的特征输出/inputs[0].tif'
        save_array_to_tif(img_array=inputs_0_np.transpose(1,2,0),out_path=tif_save_path,Band=10,Datatype=gdal.GDT_Float32)
        np.save("/data/AI-Tianlong/openmmlab/mmsegmentation/configs_new/ATL-paper-new-hiera-3/7-part-3-跨领域-crop10m-Hiera/land_use_分支的特征输出/inputs[0].npy", inputs_0_np)
        print("Saved inputs[0] to 'inputs_0.npy'")
    
        
        import pdb;pdb.set_trace()
        x_land_use_list_np = [x.detach().cpu().numpy() for x in x_land_use_list]
        np.save("/data/AI-Tianlong/openmmlab/mmsegmentation/configs_new/ATL-paper-new-hiera-3/7-part-3-跨领域-crop10m-Hiera/land_use_分支的特征输出/L1_seglogits.npy", x_land_use_list_np[0])
        np.save("/data/AI-Tianlong/openmmlab/mmsegmentation/configs_new/ATL-paper-new-hiera-3/7-part-3-跨领域-crop10m-Hiera/land_use_分支的特征输出/L2_seglogits.npy", x_land_use_list_np[1])
        np.save("/data/AI-Tianlong/openmmlab/mmsegmentation/configs_new/ATL-paper-new-hiera-3/7-part-3-跨领域-crop10m-Hiera/land_use_分支的特征输出/L3_seglogits.npy", x_land_use_list_np[2])
        
        import pdb;pdb.set_trace()
        L1_mask = np.argmax(x_land_use_list_np[0][0,:,:,:],axis=0).astype(np.uint8)  # x_land_use_list_np[0] [2,4,128,128]
        L2_mask = np.argmax(x_land_use_list_np[1][0,:,:,:],axis=0).astype(np.uint8) # [2,4,128,128]
        L3_mask = np.argmax(x_land_use_list_np[2][0,:,:,:],axis=0).astype(np.uint8) # [2,4,128,128]  # resize 到512 太那啥了

        path_ = '/data/AI-Tianlong/openmmlab/mmsegmentation/configs_new/ATL-paper-new-hiera-3/7-part-3-跨领域-crop10m-Hiera/land_use_分支的特征输出/'
        L1_mask = cv2.resize(L1_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        L2_mask = cv2.resize(L2_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        L3_mask = cv2.resize(L3_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        mask2RGB(MASK_array=L1_mask, RGB_out_path=path_, level='L1', backend='PIL')
        mask2RGB(MASK_array=L2_mask, RGB_out_path=path_, level='L2', backend='PIL')
        mask2RGB(MASK_array=L3_mask, RGB_out_path=path_, level='L3', backend='PIL')

        # Image.fromarray(L1_mask).save(os.path.join(path_, 'L1_mask.png'))
        # Image.fromarray(L2_mask).save(os.path.join(path_, 'L2_mask.png'))
        # Image.fromarray(L3_mask).save(os.path.join(path_, 'L3_mask.png'))

        import pdb;pdb.set_trace()

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)

        return losses

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`], optional): The seg data
                samples. It usually includes information such as `metainfo`
                and `gt_sem_seg`.

        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        # import pdb;pdb.set_trace()
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        seg_logits = self.inference(inputs, batch_img_metas)  # torch.Size([1, 18, 224, 224])

        return self.postprocess_result(seg_logits, data_samples)

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        x = self.extract_feat(inputs)
        return self.decode_head.forward(x)

    def slide_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                # change the image shape to patch shape
                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                # the output of encode_decode is seg logits tensor map
                # with shape [N, C, H, W]
                crop_seg_logit = self.encode_decode(crop_img, batch_img_metas)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        return seg_logits

    def whole_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference with full image.

        Args:
            inputs (Tensor): The tensor should have a shape NxCxHxW, which
                contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        seg_logits = self.encode_decode(inputs, batch_img_metas)

        return seg_logits

    def inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """Inference with slide/whole style.

        Args:
            inputs (Tensor): The input image of shape (N, 3, H, W).
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', 'pad_shape', and 'padding_size'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """
        assert self.test_cfg.get('mode', 'whole') in ['slide', 'whole'], \
            f'Only "slide" or "whole" test mode are supported, but got ' \
            f'{self.test_cfg["mode"]}.'
        ori_shape = batch_img_metas[0]['ori_shape']
        if not all(_['ori_shape'] == ori_shape for _ in batch_img_metas):
            print_log(
                'Image shapes are different in the batch.',
                logger='current',
                level=logging.WARN)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(inputs, batch_img_metas)
        else:
            seg_logit = self.whole_inference(inputs, batch_img_metas)

        return seg_logit

    def aug_test(self, inputs, batch_img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(inputs[0], batch_img_metas[0], rescale)
        for i in range(1, len(inputs)):
            cur_seg_logit = self.inference(inputs[i], batch_img_metas[i],
                                           rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(inputs)
        seg_pred = seg_logit.argmax(dim=1)
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred



def mask2RGB(
        MASK_array: str,
        RGB_out_path: str,
        level:str='L3',
        save_suffix='.png',
        backend='gdal'):
    
    reduce_zero_label = False
    L1_palette =[[146, 208, 80], [0, 100, 255], [255, 217, 102], [198, 89, 17]]                           
    L2_palette = [[112, 236, 89], [0, 150, 0], [250, 200, 0], [0, 100, 255],
                    [200, 0, 0],[255, 217, 102],[250, 200, 150],[250, 150, 0],[198, 89, 17]]   
    L3_palette=[[0,   240, 150], [150, 250, 0  ], [0,   150, 0  ], [250, 200, 0  ],
                [200, 200, 0  ], [0,   0,   200], [0,   150, 200], [150, 200, 250],
                [200, 0,   0  ], [250, 0,   150], [200, 150, 150], [250, 200, 150],
                [150, 150, 0  ], [250, 150, 150], [250, 150, 0  ], [250, 200, 250],
                [200, 150, 0  ], [200, 100, 50 ]]                           
                
    if level == 'L1':
        palette = L1_palette
    elif level == 'L2':
        palette = L2_palette
    elif level == 'L3':
        palette = L3_palette

    if reduce_zero_label:
        new_palette = [[0, 0, 0]] + palette
        # print(f"palette: {new_palette}")
    else:
        new_palette = palette
        # print(f"palette: {new_palette}")
    import numpy as np
    new_palette = np.array(new_palette)



    MASK_array[MASK_array == 255] = 0
    h,w = MASK_array.shape

    RGB_label = new_palette[MASK_array].astype(np.uint8)
    
    if backend == 'PIL':
        import os
        from PIL import Image
        output_path = os.path.join(RGB_out_path, f'{level}_rgb.{save_suffix}')
        RGB_label = Image.fromarray(RGB_label).save(output_path)
 