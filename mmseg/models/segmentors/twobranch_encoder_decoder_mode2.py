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
import os
from osgeo import gdal
from PIL import Image

from typing import Dict, Optional, Tuple, Union
from mmengine.optim import OptimWrapper
import torch

import numpy as np
from osgeo import gdal
from ATL_Tools.ATL_gdal import save_array_to_tif, save_ds_to_tif
from ATL_Tools import mkdir_or_exist
from PIL import Image
import os
import shutil

L1_vegetation_index = 0
L2_cropland_index = 0
L3_paddy_field_index = 0
L3_dry_cropland_index = 1

@MODELS.register_module()
class TwoBranch_EncoderDecoder_mode2(BaseSegmentor):
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
                 backbone: ConfigType,
                 decode_head: ConfigType,
         
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)


        # Build multi backbone 

        self.backbone = MODELS.build(backbone)        # TwoBranch_backbone_mode2
        self.decode_head = MODELS.build(decode_head)  # TwoBranch_deocde_head_mode2

        # 在这里设置不需要梯度, 成功的，在loss里面确实没有梯度，但是这里设置了eval,会被IterBaseTrainLoop给覆盖掉
        self.backbone.branch1_backbone.eval()
        for param in self.backbone.branch1_backbone.parameters():
            param.requires_grad = False

        self.decode_head.branch1_decode_head.eval()
        for param in self.decode_head.branch1_decode_head.parameters():
            param.requires_grad = False

        self.out_channels = self.decode_head.branch1_decode_head.out_channels
        self.align_corners = self.decode_head.branch1_decode_head.align_corners
        
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg


    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        # x_land_use 的list是 branch1_backbone 的输出
        # x_new_task 是两个分支进行特征交互后的输出

        x_land_use, x_new_task = self.backbone(inputs)

        return x_land_use, x_new_task


    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x_land_use, x_new_task = self.extract_feat(inputs)
        
        # 这里，不应该是分开连个seg_logits. 应该用self.decode_head的predict, 去实现有交互的输出

        seg_logits = self.decode_head.predict([x_land_use, x_new_task], batch_img_metas, self.test_cfg)

        return seg_logits
    
        # land_use_seg_logits = self.decode_head.branch1_decode_head.predict(x_land_use, batch_img_metas, self.test_cfg)
        # new_task_seg_logits = self.decode_head.branch2_decode_head.predict(x_new_task, batch_img_metas, self.test_cfg)
        # return land_use_seg_logits, new_task_seg_logits
 
        # new_task_seg_logits = self.branch2_decode_head_land_use.predict(x_new_task, batch_img_metas, self.test_cfg)



    def _decode_head_forward_train(self, 
                                   inputs: List[Tensor],
                                   data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""

        # 这里，给到TwoBranch_decode_head_mode2的loss函数
        # inputs: 这里有两个，分别为x_land_use和x_new_task的两组特征图，需要在twobranch_decode_对两个分支进行交互
        # inputs: [x_land_use, x_new_task], 分别是两个list
        # import pdb; pdb.set_trace()
                # import pdb;pdb.set_trace()

        losses = dict()
        loss_decode = self.decode_head.loss(inputs=inputs, 
                                            batch_data_samples=data_samples, 
                                            train_cfg=self.train_cfg, 
                                            test_cfg=self.test_cfg) # twobranch_decode_mode2

        # import pdb;pdb.set_trace()
        losses.update(add_prefix(loss_decode, 'branch2_decode'))
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
        self.backbone.branch1_backbone.eval()
        self.decode_head.branch1_decode_head.eval()

        # import pdb;pdb.set_trace()
        # inputs: [2, 10, 512, 512]
        # x_land_use: [2, 128, 128, 128] [2, 256, 64, 64] [2, 512, 32, 32] [2, 1024, 16, 16]
        x_land_use, x_new_task = self.extract_feat(inputs) 
                                 # 这里给出来的那个特征，应该是交互过的特征，和piip似的
                                 # 所以这里的extract_feat是twobranch_backbone_mode2的forward，并且要两个分支有交互。

        losses = dict()    # 所以我需要去定义decode_head_forward的loss + forward
        loss_decode = self._decode_head_forward_train([x_land_use, x_new_task], data_samples)
        losses.update(loss_decode)  

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
        # [1, 18, 512, 512] [1, 4, 512, 512]
        # branch1_的输出     branch2_的输出

        
        if isinstance(seg_logits, tuple) and len(seg_logits) == 2:
            barnch1_seg_logits, barnch2_seg_logits = seg_logits
            return self.postprocess_result(barnch2_seg_logits, data_samples)
            
        elif isinstance(seg_logits, Tensor):
            return self.postprocess_result(seg_logits, data_samples)
        
        else:
            TypeError(f'The type of seg_logits is {type(seg_logits)}, '
                      f'but only Tensor or tuple of Tensor is supported.')

        

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

    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        with optim_wrapper.optim_context(self):
            data = self.data_preprocessor(data, True)   #  一个是图像，一个是标签
            losses = self._run_forward(data, mode='loss')  # type: ignore
        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        optim_wrapper.update_params(parsed_losses)
        return log_vars
    
    def val_step(self, data: Union[tuple, dict, list]) -> list:
        # print_log(f'【ATL-LOG】====>>>> 自定义 val_step', logger='current')
        data = self.data_preprocessor(data, False)
        return self._run_forward(data, mode='predict')  # type: ignore

    def test_step(self, data: Union[dict, tuple, list]) -> list:
        # print_log(f'【ATL-LOG】====>>>> 自定义  test_step', logger='current')
        data = self.data_preprocessor(data, False)
        return self._run_forward(data, mode='predict')  # type: ignore


def set_requires_grad(nets, requires_grad=False):
    """Set requires_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not.
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad



