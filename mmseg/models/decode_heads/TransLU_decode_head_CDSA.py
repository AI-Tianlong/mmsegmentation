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
from mmseg.models.decode_heads.uper_head_hiera_with_gate import UPerHeadWithGate
from mmseg.models.decode_heads.uper_head_BHCCM_TransLU_CDSA import UPerHead_BHCCM_TransLU_CDSA



# For_crop_land
LCLU_L1_vegetation_index = 0      # L1: 0-vegetation, 2-water, 3-Artificial surface, 4-Bareland
LCLU_L2_cropland_index = 0        # L2: 0-cropland, 1-forest, 
LCLU_L3_paddy_field_index = 0
LCLU_L3_dry_cropland_index = 1


# @MODELS.register_module()
class TransLU_decode_head_CDSA(BaseModule):
# class TransLU_decode_head_CDSA(BaseDecodeHead):

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
                 ):
        super().__init__()

        # 直接构造
        self.branch1_decode_head = MODELS.build(branch1_decode_head)  # convnext
        self.branch2_decode_head = MODELS.build(branch2_decode_head)  # convnext

        # self.branch1_decode_head_land_use 不需要梯度
        self.branch2_decode_head.eval()
        for param in self.branch2_decode_head.parameters():
            param.requires_grad = False
        
        self.align_corners = branch1_decode_head.align_corners
        self.sampler = None
        self.ignore_index = 255

    # 这里除了传递inputs参数，还应该把batch_img_metas和标签传进来。
    # 被 Encoder_Decoder 调用
    def forward(self, inputs, batch_img_metas=None): # 这个inputs 是两个list
        
        branch1_inputs = inputs[0] # [4,128,128,128][4,256,64,64][4,512,32,32][4,1024,16,16] x_NewTask
        branch2_inputs = inputs[1] # [4,128,128,128][4,256,64,64][4,512,32,32][4,1024,16,16] x_LCLU
        
        x_LCLU_seglogits_list = self.branch2_decode_head.forward(branch2_inputs) # [[4,128,128,128][4,256,64,64][4,512,32,32][4,1024,16,16]]-->[[4,4,128,128][4,9,128,128][4,18,128,128]]
        
        # 经过softmax之后的值,非常的高,范围更保守！
        LCLU_L1_seglogits_softmax = F.softmax(x_LCLU_seglogits_list[0], dim=1)  # [2, 4,  128, 128] # 通道维度argmax？
        LCLU_L2_seglogits_softmax = F.softmax(x_LCLU_seglogits_list[1], dim=1)  # [2, 9,  128, 128]
        LCLU_L3_seglogits_softmax = F.softmax(x_LCLU_seglogits_list[2], dim=1)  # [2, 18, 128, 128]

        # 生成用于监督branch1的为标签
        L1_mask_label = torch.argmax(x_LCLU_seglogits_list[0], dim=1)
        L2_mask_label = torch.argmax(x_LCLU_seglogits_list[1], dim=1)

        L1_binary_mask_veg = (L1_mask_label == LCLU_L1_vegetation_index)
        L2_binary_mask_cropland = (L2_mask_label == LCLU_L2_cropland_index)
        
        L1_binary_mask_veg = L1_binary_mask_veg.to(torch.uint8).detach()
        L2_binary_mask_cropland = L2_binary_mask_cropland.to(torch.uint8).detach()

        LCLU_label = [L1_binary_mask_veg, L2_binary_mask_cropland]

        # import pdb;pdb.set_trace()
        # L1: 植被掩膜
        LCLU_L1_vegetation_seglogits_softmax = LCLU_L1_seglogits_softmax[:, LCLU_L1_vegetation_index, :, :]  # [2, 128, 128] 提取植被的特征图, 去作为branch1 L1的学习目标
        # L2: 耕地掩膜
        LCLU_L2_crop_seglogits_softmax = LCLU_L2_seglogits_softmax[:, LCLU_L2_cropland_index, :, :]          # [2, 128, 128] 提取耕地的特征图，去作为branch1 L2的学习目标
        # L3: xx掩膜。

        # L1-->L2-->水稻 是绝对的包含和被包含的关系
        # 植被-->耕地-->水稻、玉米、大豆。
        # 所以这里区分水稻的模型，应该去学习三个东西：
        # L1: 是植被/ 不是植被
        # L2: 是耕地/ 不是耕地
        # L3: 是目标地物 / 不是目标地物
        # L4: 水稻 / 玉米 / 大豆 / 其他 

        if isinstance(self.branch1_decode_head, UPerHead_BHCCM_TransLU_CDSA):
            LCLU_L1_softmask = LCLU_L1_vegetation_seglogits_softmax # L1 植被掩膜 # [2, 128, 128]
            LCLU_L2_softmask = LCLU_L2_crop_seglogits_softmax       # L2 耕地掩膜 # [2, 128, 128]

            LCLU_softmask_list = [LCLU_L1_softmask, LCLU_L2_softmask] #[B,128,128],[B,128,128]
            
            # import pdb; pdb.set_trace()
            softmask_and_barnch1_inputs = (LCLU_softmask_list, branch1_inputs) # tuple
            branch1_decode_head_seglogits = self.branch1_decode_head.forward(softmask_and_barnch1_inputs)  # [2, 4, 128, 128]
            
            # 所以这里是branch1的 decode_head的forward 去要去解决的
            return LCLU_label, branch1_decode_head_seglogits # 三个层级


        # debug_save_results = True
        # if debug_save_results:
        #     import pdb; pdb.set_trace()
        #     input0_data_sample = batch_img_metas[0]
        #     input0_img_path = input0_data_sample.img_path
        #     input0_label_path = input0_data_sample.seg_map_path
        #     input0_label_np = np.array(Image.open(input0_label_path))

        #     img_name = os.path.basename(input0_img_path)
        #     save_dir_path = '/data/AI-Tianlong/openmmlab/mmsegmentation/configs_new/ATL-paper-new-hiera-6-20251230-修正upernet通道/TransLU-crop10m-CDKS+CDSA/测试TransLU'
        #     save_dir_path = os.path.join(save_dir_path+'_'+img_name.split('.tif')[0])
            
        #     # copy img & RGB label
        #     mkdir_or_exist(save_dir_path)
        #     shutil.copy(input0_img_path, os.path.join(save_dir_path, img_name))
        #     mask2RGB(img_path=input0_img_path, MASK_array=input0_label_np, RGB_out_path=save_dir_path, level='crop', backend='gdal')


        #     L1_pred_mask = x_LCLU_seglogits_list[0][0].squeeze().argmax(dim=0, keepdim=True).squeeze().cpu().numpy()  # keepdim=True，保留第0维度，大小为1
        #     L2_pred_mask = x_LCLU_seglogits_list[1][0].squeeze().argmax(dim=0, keepdim=True).squeeze().cpu().numpy()  # keepdim=True，保留第0维度，大小为1
        #     L3_pred_mask = x_LCLU_seglogits_list[2][0].squeeze().argmax(dim=0, keepdim=True).squeeze().cpu().numpy()  # keepdim=True，保留第0维度，大小为1
            

        #     import cv2
        #     L1_pred_mask = cv2.resize(L1_pred_mask, dsize=(512,512), interpolation=cv2.INTER_NEAREST)
        #     L2_pred_mask = cv2.resize(L2_pred_mask, dsize=(512,512), interpolation=cv2.INTER_NEAREST)
        #     L3_pred_mask = cv2.resize(L3_pred_mask, dsize=(512,512), interpolation=cv2.INTER_NEAREST)

        #     mask2RGB(img_path=input0_img_path, MASK_array=L1_pred_mask, RGB_out_path=save_dir_path, level='L1', backend='gdal')
        #     mask2RGB(img_path=input0_img_path, MASK_array=L2_pred_mask, RGB_out_path=save_dir_path, level='L2', backend='gdal')
        #     mask2RGB(img_path=input0_img_path, MASK_array=L3_pred_mask, RGB_out_path=save_dir_path, level='L3', backend='gdal')

        #     x_lan_use_seglogits_list_np_cpu = [x.detach().cpu().numpy() for x in x_LCLU_seglogits_list]
        #     np.save(os.path.join(save_dir_path,'L1_seglogits.npy'), x_lan_use_seglogits_list_np_cpu[0])
        #     np.save(os.path.join(save_dir_path,'L2_seglogits.npy'), x_lan_use_seglogits_list_np_cpu[1])
        #     np.save(os.path.join(save_dir_path,'L3_seglogits.npy'), x_lan_use_seglogits_list_np_cpu[2])

        #     np.save(os.path.join(save_dir_path,'L1_vegetation_seglogits_softmax.npy'), LCLU_L1_seglogits_softmax.detach().cpu().numpy())
        #     np.save(os.path.join(save_dir_path,'L2_crop_seglogits_softmax.npy'), LCLU_L2_crop_seglogits_softmax.detach().cpu().numpy())

        #     import pdb; pdb.set_trace()


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
      
        # encoder_decoder.loss    --> decode.loss    里面，原始的 batch_data_samples 是有metainfo的！ 所以可以传递！
        # encoder_decoder.predict --> deocde.predict 传递的参数是 batch_img_metas, 没有metainfo，所以不要
        
        # seg_logits = self.forward(inputs)  # [2,4,128,128]          # 经过两个特征交互的。
        # 这里传递来的inputs, 其实有两部分，一个是 branch1 的 一个是 branch2的
        # import pdb; pdb.set_trace()

        # Encoder_decoder.loss() 传入的 inputs:[x_NewTasl, x_LCLU]
        if isinstance(inputs, list) and len(inputs)==2:
            
            LCLU_label, seg_logits = self.forward(inputs, batch_data_samples)  # 这里，都正常，同BHCCM, []
            losses = self.loss_by_feat(seg_logits, batch_data_samples, LCLU_label=LCLU_label) # 三个层级，去算loss
            return losses
        
        else: 
            raise TypeError
       

        
    def loss_by_feat(self, 
                     seg_logits: Tensor,
                     batch_data_samples: SampleList,
                     LCLU_label) -> dict:
        """Compute segmentation loss.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if isinstance(seg_logits, list) and len(seg_logits) == 3:
            seg_logits = seg_logits
        else:
            raise TypeError(f'请检查decode upernet forward的输出！')

        seg_label = self._stack_batch_gt(batch_data_samples)  # [B,1,512,512]
        loss = dict()
        
        # 这里需要引入branch2的输出，然后做argmax，然后作为branch1的标签

        # 原来是直接把，[2,65,128,128]---双线性插值--->[2,65,512,512]
        # 这个loss得修改呀
        # 给LCLU_label处理尺寸
        if isinstance(LCLU_label, list) and len(LCLU_label) == 2:
            for i in range(len(LCLU_label)):
                # [B, H, W] -> [B, 1, H, W] -> resize -> [B, H, W]
                LCLU_label[i] = resize(
                    input=LCLU_label[i].unsqueeze(1).float(),
                    size=seg_label.shape[2:],
                    mode='nearest').squeeze(1).long()

        if isinstance(seg_logits, list) and len(seg_logits) == 3:
            for i in range(len(seg_logits)):
                seg_logits[i] = resize(
                    input=seg_logits[i],
                    size=seg_label.shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners)
                # print(seg_logits[i].shape)
        elif isinstance(seg_logits, torch.Tensor) and seg_logits.shape[1]==sum(self.num_classes_level_list):
            seg_logits = resize(
                    input=seg_logits,
                    size=seg_label.shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners)
        else:
            raise TypeError(
                f"`seg_logits` 类型不正确，期望为长度为 3 的列表或满足指定通道数的 Tensor，"
                f"但收到类型：{type(seg_logits)}")
        
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None
            # print('走的这里')

        seg_label = seg_label.squeeze(1)  # [2,1,512,512]-->[2,512,512]


        if not isinstance(self.branch1_decode_head.loss_decode, nn.ModuleList):
            losses_decode = [self.branch1_decode_head.loss_decode]
        else:
            losses_decode = self.branch1_decode_head.loss_decode
        

        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:  # loss['atl_loss_ce'],log就打印decode.atl_loss_ce
                # import pdb; pdb.set_trace()
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logits, # seg_logits，三个
                    seg_label,  # [B,512,512] crop_label
                    LCLU_label, # LCLU_label
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
                # print(loss)
                # import pdb; pdb.set_trace()
            else:
                # pdb.set_trace()
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logits,
                    seg_label,  # [B,512,512] crop_label
                    LCLU_label, # LCLU_label
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
                # pdb.set_trace()

        # 算精度的时候
        # 融合 L1+L2+L3 三层
        if isinstance(seg_logits, list) and len(seg_logits) == 3:
            # 合并L1 L2 L3 级的推理结果
            seg_logits_L1 = seg_logits[0] # L1的特征图
            seg_logits_L2 = seg_logits[1] # L2的特征图
            seg_logits_L3 = seg_logits[2] # L3的特征图
        else:
            raise TypeError(f'seg_logits 应该是个 list ',f'但是得到了个 {type(seg_logits)}')

        # L1、L2、L3级别的精度
        
        loss['acc_seg_L1'] = accuracy(seg_logits_L1, LCLU_label[0], ignore_index=self.ignore_index)
        loss['acc_seg_L2'] = accuracy(seg_logits_L2, LCLU_label[1], ignore_index=self.ignore_index)
        loss['acc_seg_L3'] = accuracy(seg_logits_L3, seg_label, ignore_index=self.ignore_index)
        loss['acc_seg'] = accuracy(seg_logits_L3, seg_label, ignore_index=self.ignore_index)

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
        LCLU_label, seg_logits = self.forward(inputs=inputs,
                                  batch_img_metas=batch_img_metas)  # 过decode_head的forward--->[2,40,128,128]


        return self.predict_by_feat(seg_logits, batch_img_metas)   # [2,40,512,512]
   
    def predict_by_feat(self, 
                        seg_logits: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Transform a batch of output seg_logits to the input shape.  # 缩放！

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        
        seg_logits = self.branch1_decode_head.predict_by_feat(seg_logits, batch_img_metas)
        
        return seg_logits  # 输入 seg_logits的list，然后去自动处理成三个mask
    
    def jsps_inference_from_logits(self, pred_seg_logits, valid_paths: torch.Tensor):
        device = pred_seg_logits[0].device
        valid_paths = valid_paths.to(device=device, dtype=torch.long)

        logits1, logits2, logits3 = pred_seg_logits
        B, _, H, W = logits1.shape
        T = valid_paths.shape[0]

        # 1) log-prob per level: (B,C,H,W) -> (B,H,W,C)
        logp1 = F.log_softmax(logits1, dim=1).permute(0, 2, 3, 1)
        logp2 = F.log_softmax(logits2, dim=1).permute(0, 2, 3, 1)
        logp3 = F.log_softmax(logits3, dim=1).permute(0, 2, 3, 1)

        # 2) enumerate paths and sum log probs: path_scores (B,H,W,T)
        path_scores = torch.zeros((B, H, W, T), device=device, dtype=logp1.dtype)

        idx1 = valid_paths[:, 0].view(1, 1, 1, T).expand(B, H, W, T)
        idx2 = valid_paths[:, 1].view(1, 1, 1, T).expand(B, H, W, T)
        idx3 = valid_paths[:, 2].view(1, 1, 1, T).expand(B, H, W, T)

        path_scores += torch.gather(logp1, dim=-1, index=idx1)
        path_scores += torch.gather(logp2, dim=-1, index=idx2)
        path_scores += torch.gather(logp3, dim=-1, index=idx3)

        # 3) best path per pixel
        best_path_idx = path_scores.argmax(dim=-1)  # (B,H,W)

        # 4) decode per-level predictions from best path
        preds_L1 = valid_paths[best_path_idx, 0]  # (B,H,W)
        preds_L2 = valid_paths[best_path_idx, 1]
        preds_L3 = valid_paths[best_path_idx, 2]

        return preds_L1, preds_L2, preds_L3, best_path_idx, path_scores

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
    # import numpy as np
    new_palette = np.array(new_palette)

    MASK_array[MASK_array == 255] = 0
    # import pdb;pdb.set_trace()
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

 