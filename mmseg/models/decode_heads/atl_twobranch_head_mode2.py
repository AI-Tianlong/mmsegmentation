# Copyright (c) OpenMMLab. All rights reserved.
from venv import logger

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import SampleList
from ..utils import resize
from .decode_head import BaseDecodeHead
from .psp_head import PPM

from mmengine.model import BaseModule
from torch import Tensor
from typing import List, Tuple
from mmseg.utils import SampleList, ConfigType
import os
from ATL_Tools import mkdir_or_exist,find_data_list
import shutil
from mmseg.models.losses import accuracy

import numpy as np
from osgeo import gdal
from PIL import Image

from torch.nn import functional as F

from mmseg.models.decode_heads.uper_head import UPerHead


L1_vegetation_index = 0
L2_cropland_index = 0
L3_paddy_field_index = 0
L3_dry_cropland_index = 1


@MODELS.register_module()
# class TwoBranchd_decode_head_mode2(nn.Module):
class TwoBranch_decode_head_mode2(BaseModule):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    # in_index=[0, 1, 2, 3]

    def __init__(self, 
                 branch1_decode_head={},
                 branch2_decode_head={},
                 mode='xiaorong1',
                 ):
        super().__init__()
        self.mode = mode

        # 直接构造
        self.branch1_decode_head = MODELS.build(branch1_decode_head)  # convnext
        self.branch2_decode_head = MODELS.build(branch2_decode_head)  # convnext

        # self.branch1_decode_head_land_use 不需要梯度
        self.branch1_decode_head.eval()
        for param in self.branch1_decode_head.parameters():
            param.requires_grad = False

        self.align_corners = branch1_decode_head.align_corners
        self.sampler = None
        self.ignore_index = 255

        # 这里除了传递inputs参数，还应该把batch_img_metas和标签传进来。
    def forward(self, inputs, batch_img_metas=None): # 这个inputs 是两个list
        
        branch1_inputs = inputs[0]
        branch2_inputs = inputs[1]
        

        # 这里不对，我要的是特征[128,128]的，而不是[512,512]的。用forward，而不是predict
        # 通过分支1，获取相关的特征mask, 这里有问题，不应该直接获取到512,512 的mask，而应该获得128，128的mask，用forward，而不是predict
        # self.branch1_decode_head.test_output_level='L1'
        # x_land_use_seglogits_L1 = self.branch1_decode_head.predict(branch1_inputs, batch_img_metas, test_cfg) # 1, 4, 512, 512
        # self.branch1_decode_head.test_output_level='L2'
        # x_land_use_seglogits_L2 = self.branch1_decode_head.predict(branch1_inputs, batch_img_metas, test_cfg) # 1, 9, 512, 512
        # self.branch1_decode_head.test_output_level='L3'
        # x_land_use_seglogits_L3 = self.branch1_decode_head.predict(branch1_inputs, batch_img_metas, test_cfg) # 1, 18, 512, 512

        # x_lan_use_seglogits_list = [x_land_use_seglogits_L1, x_land_use_seglogits_L2, x_land_use_seglogits_L3]
        
        # import pdb; pdb.set_trace()
        x_lan_use_seglogits_list, _ = self.branch1_decode_head.forward(branch1_inputs) # 1, 4, 128, 128 :embedding:[2,256,16,16]
        # [2,128,128,128][2,256,64,64][2,512,32,32][2,1024,16,16] --> [2, 4, 128, 128] [2, 9, 128, 128] [2, 18, 128, 128]


        # 经过softmax之后的值,非常的高,范围更保守！
        L1_seglogits_softmax = F.softmax(x_lan_use_seglogits_list[0], dim=1)  # [2, 4,  128, 128]
        L2_seglogits_softmax = F.softmax(x_lan_use_seglogits_list[1], dim=1)  # [2, 9,  128, 128]
        L3_seglogits_softmax = F.softmax(x_lan_use_seglogits_list[2], dim=1)  # [2, 18, 128, 128]

        # import pdb;pdb.set_trace() # 从land use 中，取出 符合那啥的值。
        L1_vegetation_seglogits_softmax = L1_seglogits_softmax[:, L1_vegetation_index, :, :]  # [2, 128, 128]
        L2_crop_seglogits_softmax = L2_seglogits_softmax[:, L2_cropland_index, :, :]          # [2, 128, 128]
        # mask 有了，如何嵌入到新的任务中去呢？

        # import pdb; pdb.set_trace()
        # 这里可以作为一个消融，直接用branch2_decode_head的forward去获得输出。
        
        if self.mode == 'xiaorong1':  # head没有交互，只用branch2_decode_head的特征
            branch2_decode_head_seglogits = self.branch2_decode_head.forward(branch2_inputs)  # [2, 4, 128, 128]
            return branch2_decode_head_seglogits
        
        elif self.mode == 'xiaorong2':
            if isinstance(self.branch2_decode_head, UPerHead):  # 如果是UperHead的话
               branch2_decode_head_seglogits = self.branch2_decode_head.forward(branch2_inputs)  # [2, 4, 128, 128]
               return branch2_decode_head_seglogits



        # 这两个mask是很重要的！
        # 然后很重要的就是 两个模型之间的特征。需要交互，来促进学习。

        # 模型之间的交互，还是用可形变注意力？
        # test1：
        # 需要一个interactions，类似于TwoBranch的交互


        # branch2_new_task_output = self.branch2_decode_head.forward(branch2_inputs)
        
        # 在这里，应该把branch1 和 branch2 的结构拿出来，像piip一样。

        debug_save_results = False
        if debug_save_results:
            import pdb; pdb.set_trace()
            input0_data_sample = batch_img_metas[0]
            input0_img_path = input0_data_sample['img_path']
            input0_label_path = input0_data_sample['seg_map_path']
            input0_label_np = np.array(Image.open(input0_label_path))

            img_name = os.path.basename(input0_img_path)
            save_dir_path = '/data/AI-Tianlong/openmmlab/mmsegmentation/configs_new/ATL-paper-new-hiera-3/7-part-3-跨领域-crop10m-Hiera/特征输出-2branch-loss'
            save_dir_path = os.path.join(save_dir_path+'_'+img_name.split('黑龙江省_')[1].split('.tif')[0])
            
            # copy img & RGB label
            mkdir_or_exist(save_dir_path)
            shutil.copy(input0_img_path, os.path.join(save_dir_path, img_name))
            mask2RGB(img_path=input0_img_path, MASK_array=input0_label_np, RGB_out_path=save_dir_path, level='crop', backend='gdal')


            L1_pred_mask = x_lan_use_seglogits_list[0].squeeze().argmax(dim=0, keepdim=True).squeeze().cpu().numpy()  # keepdim=True，保留第0维度，大小为1
            L2_pred_mask = x_lan_use_seglogits_list[1].squeeze().argmax(dim=0, keepdim=True).squeeze().cpu().numpy()  # keepdim=True，保留第0维度，大小为1
            L3_pred_mask = x_lan_use_seglogits_list[2].squeeze().argmax(dim=0, keepdim=True).squeeze().cpu().numpy()  # keepdim=True，保留第0维度，大小为1
            
            mask2RGB(img_path=input0_img_path, MASK_array=L1_pred_mask, RGB_out_path=save_dir_path, level='L1', backend='gdal')
            mask2RGB(img_path=input0_img_path, MASK_array=L2_pred_mask, RGB_out_path=save_dir_path, level='L2', backend='gdal')
            mask2RGB(img_path=input0_img_path, MASK_array=L3_pred_mask, RGB_out_path=save_dir_path, level='L3', backend='gdal')

            x_lan_use_seglogits_list_np_cpu = [x.detach().cpu().numpy() for x in x_lan_use_seglogits_list]
            np.save(os.path.join(save_dir_path,'L1_seglogits.npy'), x_lan_use_seglogits_list_np_cpu[0])
            np.save(os.path.join(save_dir_path,'L2_seglogits.npy'), x_lan_use_seglogits_list_np_cpu[1])
            np.save(os.path.join(save_dir_path,'L3_seglogits.npy'), x_lan_use_seglogits_list_np_cpu[2])

            np.save(os.path.join(save_dir_path,'L1_vegetation_seglogits_softmax.npy'), L1_vegetation_seglogits_softmax.detach().cpu().numpy())
            np.save(os.path.join(save_dir_path,'L2_crop_seglogits_softmax.npy'), L2_crop_seglogits_softmax.detach().cpu().numpy())

            import pdb; pdb.set_trace()

        
        
        # return branch1_land_use_output, branch2_new_task_output


    def loss(self, 
             inputs: Tuple[Tensor], 
             batch_data_samples: SampleList,
             train_cfg: ConfigType,
             test_cfg: ConfigType) -> dict:
        """Calculate losses from a list of inputs and data samples.

        Args:
            inputs (list[Tensor]): List of input tensors.
            data_samples (list[BaseDataElement], optional): List of
                data samples. Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # decode_head的loss函数:
        #    (1) seg_logits = self.forward(inputs)  # [2,40,128,128]-->得到一个特征图
        #    (2) losses = self.loss_by_feat(seg_logits, batch_data_samples)
        # 这里，最后返回的是那个 头的seglogits就行，需要去拆解upernet的模块
        # import pdb; pdb.set_trace() # TwoBranch_decode_head_mode2的forward函数
        # import pdb; pdb.set_trace()
        # encoder.loss-->decode.loss里面，原始的           batch_data_samples 是有metainfo的！ 所以可以传递！
        # encoder.predict --> deocde.predict 传递的参数是  batch_img_metas, 没有metainfo，所以不要
        seg_logits = self.forward(inputs)  # [2,4,128,128]
        losses = self.loss_by_feat(seg_logits, batch_data_samples) 
        return losses
        

        # 这里有一个loss2, 可以去算一下branch2_decode_head 和标签的loss，不经过任何的辅助
        # losses = self.branch2_decode_head.loss(branch2_inputs, batch_data_samples, train_cfg)
        # return losses

    def loss_by_feat(self, seg_logits: Tensor,
                     batch_data_samples: SampleList) -> dict:
        """Compute segmentation loss.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # seg_logits: [2,65,128,128]
        # [2,1,512,512]
        seg_label = self._stack_batch_gt(batch_data_samples)  # [2,1,512,512]
        loss = dict()
        # 原来是直接把，[2,65,128,128]---双线性插值--->[2,65,512,512]
        seg_logits = resize(
            input=seg_logits,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None
            # print('走的这里')

        seg_label = seg_label.squeeze(1)  # [2,1,512,512]-->[2,512,512]

        # import pdb
        # pdb.set_trace()


        if not isinstance(self.branch2_decode_head.loss_decode, nn.ModuleList):
            losses_decode = [self.branch2_decode_head.loss_decode]
        else:
            losses_decode = self.branch2_decode_head.loss_decode
        
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:  # loss['atl_loss_ce'],log就打印decode.atl_loss_ce
                
                # import pdb; pdb.set_trace()
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
                # print(loss)
                # import pdb; pdb.set_trace()
            else:
                # pdb.set_trace()
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
                # pdb.set_trace()
        # pdb.set_trace()
        loss['acc_seg'] = accuracy(
            seg_logits, seg_label, ignore_index=self.ignore_index)
        # pdb.set_trace()
        return loss
    
    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples # 这里的 gt_sem_seg, 用来去坐loss损失
        ]
        # [2,1,512,512]
        return torch.stack(gt_semantic_segs, dim=0)


    def predict(self, inputs: Tuple[Tensor], 
                batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tensor:
        """Forward function for prediction.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_img_metas (dict): List Image info where each dict may also
                contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        # import pdb; pdb.set_trace()
        seg_logits = self.forward(inputs=inputs,
                                  batch_img_metas=batch_img_metas)  # 过decode_head的forward--->[2,40,128,128]


        return self.predict_by_feat(seg_logits, batch_img_metas)   # [2,40,512,512]

    def predict_by_feat(self, seg_logits: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Transform a batch of output seg_logits to the input shape.  # 缩放！

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tensor: Outputs segmentation logits map.
        """

        if isinstance(batch_img_metas[0]['img_shape'], torch.Size):
            # slide inference
            size = batch_img_metas[0]['img_shape']
        elif 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape'][:2]
        else:
            size = batch_img_metas[0]['img_shape']
            
        seg_logits = resize(
            input=seg_logits,
            size=size,
            mode='bilinear',
            align_corners=self.align_corners)
        return seg_logits


def mask2RGB(
        img_path: str,
        MASK_array: str,
        RGB_out_path: str,
        level:str='L3',
        save_suffix='.tif',
        backend='gdal'):

    reduce_zero_label = False
    img_name = os.path.basename(img_path)
    
    L1_palette = [[146, 208, 80], [0,  100,  255], [255, 217, 102], [198, 89, 17]]                           
    L2_palette = [[112, 236, 89], [0,  150,  0  ], [250, 200, 0  ], [0, 100, 255],
                  [200, 0,   0],  [255, 217, 102], [250, 200, 150], [250, 150, 0],
                  [198, 89, 17]]   
    L3_palette=[[0,   240, 150], [150, 250, 0  ], [0,   150, 0  ], [250, 200, 0  ],
                [200, 200, 0  ], [0,   0,   200], [0,   150, 200], [150, 200, 250],
                [200, 0,   0  ], [250, 0,   150], [200, 150, 150], [250, 200, 150],
                [150, 150, 0  ], [250, 150, 150], [250, 150, 0  ], [250, 200, 250],
                [200, 150, 0  ], [200, 100, 50 ]]                  
    crop_palette = [[255, 255, 255], [0, 200, 250], [250, 200, 0], [150, 150, 250]]

    if level == 'L1':
        palette = L1_palette
    elif level == 'L2':
        palette = L2_palette
    elif level == 'L3':
        palette = L3_palette
    elif level == 'crop':   
        palette = crop_palette

    if reduce_zero_label:
        new_palette = [[0, 0, 0]] + palette
        # print(f"palette: {new_palette}")
    else:
        new_palette = palette
        # print(f"palette: {new_palette}")
    import numpy as np
    new_palette = np.array(new_palette)

    MASK_array[MASK_array == 255] = 0
    h, w = MASK_array.shape

    RGB_label = new_palette[MASK_array].astype(np.uint8)
    
    if backend == 'PIL':

        output_path = os.path.join(RGB_out_path, f'{level}_rgb_{img_name}')
        RGB_label = Image.fromarray(RGB_label).save(output_path)

    elif backend == 'gdal':
        output_path = os.path.join(RGB_out_path, f'{level}_rgb_{img_name}')
        driver = gdal.GetDriverByName('GTiff')
        RGB_label_gdal = driver.Create(output_path, w, h, 3, gdal.GDT_Byte)

        RGB_label_gdal.GetRasterBand(1).WriteArray(RGB_label[:,:,0])
        RGB_label_gdal.GetRasterBand(2).WriteArray(RGB_label[:,:,1])
        RGB_label_gdal.GetRasterBand(3).WriteArray(RGB_label[:,:,2])

        img_gdal = gdal.Open(img_path, gdal.GA_ReadOnly)
        assert  img_path is not None, f"无法打开 {img_path}"

        trans = img_gdal.GetGeoTransform()
        proj = img_gdal.GetProjection()

        RGB_label_gdal.SetGeoTransform(trans)
        RGB_label_gdal.SetProjection(proj)

        RGB_label_gdal = None

 