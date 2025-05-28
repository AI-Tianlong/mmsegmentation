# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .cross_entropy_loss import CrossEntropyLoss, cross_entropy
from .utils import get_class_weight, weight_reduce_loss

# S2_5B 数据集，分为L1、L2、L3三级标签，共21类
# L1: 包含5类, 0-4
# L2: 包含11类, 0-10
# L3: 包含21类, 0-20

# reduce_zero_label 后的值 
# L1_L2map = [[L2的index，在L1为一组][][]]
# L2_L3map = [[L3的index，在L2是一组]]
L1_L2map = [[0,1,2],[3],[4,5,6,7],[8]]
L2_L3map = [[0,1],[2],[3,4],[5,6,7],[8],[9,10],[11,12],[13,14,15,16],[17]]

L3_L3map = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]

MM_5B_18_hiera_structure = dict(
    # class_L1_{L1中的标签号}_{L1中的标签名称}=[L3级标签的值]
    Classes_Map_L1=dict(
        class_L1_0_Vegetation=[0, 1, 2, 3, 4],
        class_L1_1_Water=[5,6,7],
        class_L1_2_Artificial_surface=[8,9,10,11,12,13,14,15,16],
        class_L1_3_Bare_land=[17],
        ),
    # class_L2_{L1级标签中的标签值}_{L2级标签中的标签值}_{L2级标签中的标签名称}=[L3级标签中的值]
    Classes_Map_L2=dict(
        class_L2_0_Crop_land=[0, 1],
        class_L2_1_Forest=[2],
        class_L2_2_Grassland=[3,4],
        class_L2_3_Water=[5,6,7],
        class_L2_4_Factory_Shopping_malls=[8],
        class_L2_5_Residence=[9,10],
        class_L2_6_Public_area=[11,12],
        class_L2_7_Transportation_infrastructure=[13,14,15,16],
        class_L2_8_Bare_land=[17],
        ),
    # class_L3_{L1级标签中的标签值}_{L2级标签中的标签值}_{L3级标签中的标签值}_{L3级标签中的标签名称}
    Classes_Map_L3=dict(
        class_L3_0_Paddy_field=[0],
        class_L3_1_Dry_cropland=[1],
        class_L3_2_Forest=[2],
        class_L3_3_Natural_meadow=[3],
        class_L3_4_Artificial_meadow=[4],
        class_L3_5_River=[5],
        class_L3_6_Lake=[6],
        class_L3_7_Pond=[7],
        class_L3_8_Factory_shopping_malls=[8],
        class_L3_9_Urban_residential=[9],
        class_L3_10_Rural_residential=[10],
        class_L3_11_Stadium=[11],
        class_L3_12_Park_Square=[12],
        class_L3_13_Road=[13],
        class_L3_14_Overpass=[14],
        class_L3_15_Railway_station=[15],
        class_L3_16_Airport=[16],
        class_L3_17_Bare_land=[17]
        )
    )


def convert_low_level_label_to_High_level(label, classes_map):
    """Convert low level label to High level label.
        e.g.:
          convert L3 label (num_classes=22)  to
          L2 label (num_classes=12) or L1 label (num_classes=6)

    Args:
        label (Tensor): L3 label.  label.shape:[2, 512, 512]
        classes_map (dict): Classes map.

    Returns:
        Tensor: Lx label.
    """

    label_list = list()  # 转换 L3 (low level) --> L1 L2 (hgih level)
    for _, high_level_dict in list(classes_map.items())[:-1]:
        high_level_label = torch.zeros_like(label).fill_(255)  # [2,512,512] 255 is the ignore index #因为用了like，所以也在GPU上
        for high_level_label_value, high_level_key in enumerate(high_level_dict): # For L1:0 1 2 3 4
            low_level_label_list = high_level_dict[high_level_key]
            for low_level_label in low_level_label_list:
                high_level_label[label == low_level_label] = high_level_label_value

        label_list.append(high_level_label)
    label_list.append(label)  # add L3 label to label List
    return label_list  # [L1级label, L2级label, L3级label] #tensor [2,512,512][2,512,512][2,512,512]


# 消融实验--------------> 只保留三个celoss 
@MODELS.register_module()
class HCC_LOSS(nn.Module):

    def __init__(self,
                 num_classes,
                 use_sigmoid=False,
                 loss_name = 'loss_HCC',
                 loss_weight=1.0,
                 ignore_index=255,
                 mode='HCC'):
        super().__init__()
        self.num_classes = num_classes # [4,9,18] list
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index  # 应该都是255了
        # self.tree_triplet_loss = TreeTripletLoss(ignore_index = self.ignore_index)
        # self.tree_triplet_loss = Focal_Tree_Min_Loss(ignore_index = self.ignore_index)
        self.cross_entropy_loss = CrossEntropyLoss(loss_name='loss_HCC_ce')
        self._loss_name = loss_name
        self.mode = mode
    
    def forward(self,
                pred_seg_logits,   # [2,34,128,128] [5+10+19]
                label,
                ignore_index=-100,
                **kwargs):
        
        # import pdb; pdb.set_trace()

        if isinstance(pred_seg_logits, list) and len(pred_seg_logits) == 3:
            pred_seg_logits = pred_seg_logits
        
        else:   
            raise TypeError(f'pred_seg_logits 应该是个 list, 但却得到了{type(pred_seg_logits)}')
        
        # 将L3级的标签，转换为L1、L2、L3 用来计算 loss 值
        hiera_label_list = convert_low_level_label_to_High_level(label, MM_5B_18_hiera_structure)


        # ====================== 2025 年 5 月 27 日的新实验 ====================
        
        # loss 消融1
        if self.mode == '3XCE':
            self._loss_name = 'loss_3xCE'
            ce_loss_L1 = self.cross_entropy_loss(pred_seg_logits[0],
                                             hiera_label_list[0],
                                             weight=None,
                                             ignore_index=self.ignore_index)

            ce_loss_L2 = self.cross_entropy_loss(pred_seg_logits[1],
                                                hiera_label_list[1],
                                                weight=None,
                                                ignore_index=self.ignore_index)

            ce_loss_L3 = self.cross_entropy_loss(pred_seg_logits[2],
                                                hiera_label_list[2],
                                                weight=None,
                                                ignore_index=self.ignore_index)
            loss = ce_loss_L1 + ce_loss_L2 + ce_loss_L3  
        
        # loss 消融2 
        elif self.mode == 'HCC':
            w1,w2,w3 = 1.0, 1.0, 1.0
            ce_loss_L1 = self.cross_entropy_loss(pred_seg_logits[0],
                                                hiera_label_list[0],
                                                weight=None,
                                                ignore_index=self.ignore_index)
            ce_loss_L2 = self.cross_entropy_loss(pred_seg_logits[1],
                                                hiera_label_list[1],
                                                weight=None,
                                                ignore_index=self.ignore_index)
            ce_loss_L3 = self.cross_entropy_loss(pred_seg_logits[2],
                                                hiera_label_list[2],
                                                weight=None,
                                                ignore_index=self.ignore_index)
            loss_ce = w1*ce_loss_L1 + w2*ce_loss_L2 + w3*ce_loss_L3  
        
            
            # L_TK

            # 将255暂时替换为0以便one-hot（因为one_hot不支持255作为类索引）
            # 最后再计算loss的时候，把这个掩掉就行
            valid_mask  = (label != self.ignore_index)
            hiera_label_list_255to0 = [gt.clone() for gt in hiera_label_list]
            hiera_label_list_255to0[0][~valid_mask] = 0
            hiera_label_list_255to0[1][~valid_mask] = 0
            hiera_label_list_255to0[2][~valid_mask] = 0
            # one-hot编码
            
            hiera_label_one_hot = []
            for i in range(len(hiera_label_list)):
                hiera_label_one_hot.append(F.one_hot(hiera_label_list_255to0[i], num_classes=self.num_classes[i]).float()) 
            
            # \hat{Y}_i
            Y_label = torch.cat([hiera_label_one_hot[0], hiera_label_one_hot[1], hiera_label_one_hot[2]], dim=3) #[2,640,640,31]
            Y_label = Y_label.permute(0, 3, 1, 2)  # [2,31,640,640] # [batch_size, num_classes, height, width]
            
            Y_pred = torch.cat([pred_seg_logits[0], pred_seg_logits[1], pred_seg_logits[2]], dim=1)
            Y_pred = F.log_softmax(Y_pred, dim=1) 

            kl = F.kl_div(Y_pred, Y_label.float(), reduction='none') # 算完怎么还是[2,31,640,640]
            kl = kl.sum(dim=1)    # [2,31,640,640]-->[2,640,640] #一个像素一个KL散度                                   # (N,)
            
            loss_tk = kl[valid_mask].mean()

            alpha = 0.5
            loss = loss_ce + alpha * loss_tk

        return loss*self.loss_weight 
    


    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
