# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from collections import OrderedDict
from typing import Any, List, Optional, Sequence, Union, Dict

import logging
import numpy as np
import torch
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist
from PIL import Image
from prettytable import PrettyTable
from torch import Tensor
from mmengine.structures import BaseDataElement



from mmseg.registry import METRICS
from mmengine.dist import (broadcast_object_list, collect_results,
                           is_main_process)

from mmseg.models.losses.hcc_loss import convert_low_level_label_to_High_level, MM_5B_18_hiera_structure

L1_classes_name = ('Vegetation', ' Water', 'Artificial_surface', 'Bare_land')

L2_classes_name = ('Crop_land', 'Forest', 'Grassland', 'Water', 'Factory_Shopping_malls', 
                   'Residence','Public_area','Transportation','Bare_land')

L3_classes_name = ('Paddy field', 'Other Field', 'Forest', 'Natural meadow',
                   'Artificial meadow', 'River', 'Lake', 'Pond',
                   'Factory-Storage-Shopping malls', 'Urban residential',
                   'Rural residential', 'Stadium', 'Park Square', 'Road',
                   'Overpass', 'Railway station', 'Airport', 'Bare land')


@METRICS.register_module()
class IoUMetric_HSM(BaseMetric):
    """IoU evaluation metric.

    Args:
        level (str): The level of the metric. Default: L3. L1/L2/L3
        ignore_index (int): Index that will be ignored in evaluation.
            Default: 255.
        iou_metrics (list[str] | str): Metrics to be calculated, the options
            includes 'mIoU', 'mDice' and 'mFscore'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        beta (int): Determines the weight of recall in the combined score.
            Default: 1.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        output_dir (str): The directory for output prediction. Defaults to
            None.
        format_only (bool): Only format result for results commit without
            perform evaluation. It is useful when you want to save the result
            to a specific format and submit it to the test server.
            Defaults to False.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    def __init__(self,
                 baseline_or_HSM: str = 'baseline', # `baseline` or `HSM`
                 test_output_level: str = 'L3',
                 num_classes_list: List[int] = [4, 9, 18],
                 ignore_index: int = 255,
                 iou_metrics: List[str] = ['mIoU'],
                 nan_to_num: Optional[int] = None,
                 beta: int = 1,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.baseline_or_HSM = baseline_or_HSM
        self.test_output_level = test_output_level
        self.num_classes_list = num_classes_list

        self.L1_results = []
        self.L2_results = []
        

        self.ignore_index = ignore_index
        self.metrics = iou_metrics
        self.nan_to_num = nan_to_num
        self.beta = beta
        self.output_dir = output_dir
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)
        self.format_only = format_only
        
    def convert_baseline_L3_to_L1_L2(self, L3_pred_label, test_output_level):
        pred_label_list = convert_low_level_label_to_High_level(L3_pred_label, MM_5B_18_hiera_structure)
        if test_output_level == 'L3':
            pred_label = pred_label_list[2]
        elif test_output_level == 'L2':
            pred_label = pred_label_list[1]
        elif test_output_level == 'L1':
            pred_label = pred_label_list[0]
        return pred_label
    
    def baseline_L2_process(self, data_samples: Sequence[dict], level:str='L2') -> None:
        for data_sample in data_samples:    
            # 根据uperhead的输出，这里是L1、L2、L3
            pred_label = data_sample['pred_sem_seg']['data'].squeeze()  #经过Encode_Decoder的predict的结果 [594,594]
            pred_label = self.convert_baseline_L3_to_L1_L2(pred_label, level)
            label = data_sample['gt_sem_seg']['data'].squeeze().to(pred_label)
            # import pdb; pdb.set_trace()
            label_list = convert_low_level_label_to_High_level(label, MM_5B_18_hiera_structure)

            if level == 'L3':
                num_classes = self.num_classes_list[2]
                self.dataset_meta['classes'] = L3_classes_name
                label = label_list[2] # 仅输出融合后L3的特征图
            elif level == 'L2':
                num_classes = self.num_classes_list[1]
                self.dataset_meta['classes'] = L2_classes_name
                label = label_list[1]
            elif level == 'L1':
                num_classes = self.num_classes_list[0]
                self.dataset_meta['classes'] = L1_classes_name
                label = label_list[0]
            self.L2_results.append(self.intersect_and_union(pred_label, label, num_classes, self.ignore_index))
    
    def baseline_process(self, data_samples: Sequence[dict], level:str='L1') -> None:
        for data_sample in data_samples:    
            # 根据uperhead的输出，这里是L1、L2、L3
            pred_label = data_sample['pred_sem_seg']['data'].squeeze()  #经过Encode_Decoder的predict的结果 [594,594]
            pred_label = self.convert_baseline_L3_to_L1_L2(pred_label, level)
            label = data_sample['gt_sem_seg']['data'].squeeze().to(pred_label)
            # import pdb; pdb.set_trace()
            label_list = convert_low_level_label_to_High_level(label, MM_5B_18_hiera_structure)

            if level == 'L3':
                num_classes = self.num_classes_list[2]
                self.dataset_meta['classes'] = L3_classes_name
                label = label_list[2] # 仅输出融合后L3的特征图
            elif level == 'L2':
                num_classes = self.num_classes_list[1]
                self.dataset_meta['classes'] = L2_classes_name
                label = label_list[1]
            elif level == 'L1':
                num_classes = self.num_classes_list[0]
                self.dataset_meta['classes'] = L1_classes_name
                label = label_list[0]

            if level == 'L2':
                self.L2_results.append(self.intersect_and_union(pred_label, label, num_classes, self.ignore_index))
            elif level == 'L1':
                self.L1_results.append(self.intersect_and_union(pred_label, label, num_classes, self.ignore_index))
    
    
    def HSM_process(self, data_samples: Sequence[dict], level:str='L1') -> None:
        for data_sample in data_samples:    
            # 根据uperhead的输出，这里是L1、L2、L3
            pred_label = data_sample[f'pred_sem_seg_{level}']['data'].squeeze()  #经过Encode_Decoder的predict的结果 [594,594]
            label = data_sample['gt_sem_seg']['data'].squeeze().to(pred_label)
            # import pdb; pdb.set_trace()
            label_list = convert_low_level_label_to_High_level(label, MM_5B_18_hiera_structure)

            if level == 'L3':
                num_classes = self.num_classes_list[2]
                self.dataset_meta['classes'] = L3_classes_name
                label = label_list[2] # 仅输出融合后L3的特征图
            elif level == 'L2':
                num_classes = self.num_classes_list[1]
                self.dataset_meta['classes'] = L2_classes_name
                label = label_list[1]
            elif level == 'L1':
                num_classes = self.num_classes_list[0]
                self.dataset_meta['classes'] = L1_classes_name
                label = label_list[0]

            if level == 'L2':
                self.L2_results.append(self.intersect_and_union(pred_label, label, num_classes, self.ignore_index))
            elif level == 'L1':
                self.L1_results.append(self.intersect_and_union(pred_label, label, num_classes, self.ignore_index))


    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        # 先把L2和L1处理掉
        if self.baseline_or_HSM == 'baseline': # 用L3去聚合出L2和L1
            self.baseline_process(data_samples=data_samples, level='L1')  # 产生 self.results1 
            self.baseline_process(data_samples=data_samples, level='L2')  # 产生 self.results2
        elif self.baseline_or_HSM == 'HSM':  # 用实际的L3 L2 L1 去计算
            self.HSM_process(data_samples=data_samples, level='L1')  # 产生 self.results1 
            self.HSM_process(data_samples=data_samples, level='L2')  # 产生 self.results2
           
            # data_samples: reduce_zero_label / img_path / seg_map_path / ori_shape / img_shape
            #  gt_sem_seg
            # seg_logits_L1 / seg_logits_L2/ seg_logits_L3 / seg_logits
            # pred_sem_seg_L1 / pred_sem_seg_L2 / pred_sem_seg_L3 / pred_sem_seg


        # For 正常的流程，去输出去。
        num_classes = len(self.dataset_meta['classes'])
        for data_sample in data_samples:
            # 根据uperhead的输出，这里是L1、L2、L3
            pred_label = data_sample['pred_sem_seg']['data'].squeeze()  #经过Encode_Decoder的predict的结果 [594,594]
            if self.baseline_or_HSM == 'baseline':
                pred_label = self.convert_baseline_L3_to_L1_L2(pred_label, self.test_output_level)
            
            # format_only always for test dataset without ground truth
            if not self.format_only:
                label = data_sample['gt_sem_seg']['data'].squeeze().to(pred_label)
                # import pdb; pdb.set_trace()
                label_list = convert_low_level_label_to_High_level(label, MM_5B_18_hiera_structure)
                
                if self.test_output_level is None:
                    self.test_output_level = 'L3'
                if self.test_output_level == 'L3':
                    num_classes = self.num_classes_list[2]
                    self.dataset_meta['classes'] = L3_classes_name
                    label = label_list[2] # 仅输出融合后L3的特征图
                elif self.test_output_level == 'L2':
                    num_classes = self.num_classes_list[1]
                    self.dataset_meta['classes'] = L2_classes_name
                    label = label_list[1]
                elif self.test_output_level == 'L1':
                    num_classes = self.num_classes_list[0]
                    self.dataset_meta['classes'] = L1_classes_name
                    label = label_list[0]
                
                # import pdb;pdb.set_trace()
                # 其实在这里处理这个label就行。
                self.results.append(self.intersect_and_union(pred_label, label, num_classes, self.ignore_index))
                
                # The intersection of prediction and ground truth histogram on all classes.
                # The union of prediction and ground truth histogram on all classes.
                # The prediction histogram on all classes.
                # The ground truth histogram on all classes.
                
            # import pdb; pdb.set_trace()
            # format_result
            if self.output_dir is not None:
                basename = osp.splitext(osp.basename(
                    data_sample['img_path']))[0]
                png_filename = osp.abspath(
                    osp.join(self.output_dir, f'{basename}.png'))
                output_mask = pred_label.cpu().numpy()
                # The index range of official ADE20k dataset is from 0 to 150.
                # But the index range of output is from 0 to 149.
                # That is because we set reduce_zero_label=True.
                if data_sample.get('reduce_zero_label', False):
                    output_mask = output_mask + 1
                output = Image.fromarray(output_mask.astype(np.uint8))
                output.save(png_filename)


    # 这里去打印表格
    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
                mRecall.
        """
        # import pdb;pdb.set_trace()
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()
        # convert list of tuples to tuple of lists, e.g.
        # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
        # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
        results = tuple(zip(*results))
        assert len(results) == 4

        total_area_intersect = sum(results[0])
        total_area_union = sum(results[1])
        total_area_pred_label = sum(results[2])
        total_area_label = sum(results[3])
        ret_metrics = self.total_area_to_metrics(
            total_area_intersect, total_area_union, total_area_pred_label,
            total_area_label, self.metrics, self.nan_to_num, self.beta)

        class_names = self.dataset_meta['classes']

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        metrics = dict()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                metrics[key] = val
            else:
                metrics['m' + key] = val

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)

        return metrics

    @staticmethod
    def intersect_and_union(pred_label: torch.tensor, 
                            label: torch.tensor,
                            num_classes: int, 
                            ignore_index: int):
        """Calculate Intersection and Union.

        Args:
            pred_label (torch.tensor): Prediction segmentation map
                or predict result filename. The shape is (H, W).
            label (torch.tensor): Ground truth segmentation map
                or label filename. The shape is (H, W).
            num_classes (int): Number of categories.
            ignore_index (int): Index that will be ignored in evaluation.

        Returns:
            torch.Tensor: The intersection of prediction and ground truth
                histogram on all classes.
            torch.Tensor: The union of prediction and ground truth histogram on
                all classes.
            torch.Tensor: The prediction histogram on all classes.
            torch.Tensor: The ground truth histogram on all classes.
        """
        # # 临时关闭deterministic模式
        # torch.use_deterministic_algorithms(False)

        mask = (label != ignore_index)
        pred_label = pred_label[mask]
        label = label[mask]

        intersect = pred_label[pred_label == label]
        area_intersect = torch.histc(
            intersect.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_pred_label = torch.histc(
            pred_label.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_label = torch.histc(
            label.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_union = area_pred_label + area_label - area_intersect
        
        # # 临时关闭deterministic模式
        # torch.use_deterministic_algorithms(True)
        return area_intersect, area_union, area_pred_label, area_label

    @staticmethod
    def total_area_to_metrics(total_area_intersect: np.ndarray,
                              total_area_union: np.ndarray,
                              total_area_pred_label: np.ndarray,
                              total_area_label: np.ndarray,
                              metrics: List[str] = ['mIoU'],
                              nan_to_num: Optional[int] = None,
                              beta: int = 1):
        """Calculate evaluation metrics
        Args:
            total_area_intersect (np.ndarray): The intersection of prediction
                and ground truth histogram on all classes.
            total_area_union (np.ndarray): The union of prediction and ground
                truth histogram on all classes.
            total_area_pred_label (np.ndarray): The prediction histogram on
                all classes.
            total_area_label (np.ndarray): The ground truth histogram on
                all classes.
            metrics (List[str] | str): Metrics to be evaluated, 'mIoU' and
                'mDice'.
            nan_to_num (int, optional): If specified, NaN values will be
                replaced by the numbers defined by the user. Default: None.
            beta (int): Determines the weight of recall in the combined score.
                Default: 1.
        Returns:
            Dict[str, np.ndarray]: per category evaluation metrics,
                shape (num_classes, ).
        """

        def f_score(precision, recall, beta=1):
            """calculate the f-score value.

            Args:
                precision (float | torch.Tensor): The precision value.
                recall (float | torch.Tensor): The recall value.
                beta (int): Determines the weight of recall in the combined
                    score. Default: 1.

            Returns:
                [torch.tensor]: The f-score value.
            """
            score = (1 + beta**2) * (precision * recall) / (
                (beta**2 * precision) + recall)
            return score

        if isinstance(metrics, str):
            metrics = [metrics]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metrics).issubset(set(allowed_metrics)):
            raise KeyError(f'metrics {metrics} is not supported')

        all_acc = total_area_intersect.sum() / total_area_label.sum()
        ret_metrics = OrderedDict({'aAcc': all_acc})
        for metric in metrics:
            if metric == 'mIoU':
                iou = total_area_intersect / total_area_union
                acc = total_area_intersect / total_area_label
                ret_metrics['IoU'] = iou
                ret_metrics['Acc'] = acc
            elif metric == 'mDice':
                dice = 2 * total_area_intersect / (
                    total_area_pred_label + total_area_label)
                acc = total_area_intersect / total_area_label
                ret_metrics['Dice'] = dice
                ret_metrics['Acc'] = acc
            elif metric == 'mFscore':
                precision = total_area_intersect / total_area_pred_label
                recall = total_area_intersect / total_area_label
                f_value = torch.tensor([
                    f_score(x[0], x[1], beta) for x in zip(precision, recall)
                ])
                ret_metrics['Fscore'] = f_value
                ret_metrics['Precision'] = precision
                ret_metrics['Recall'] = recall

        ret_metrics = {
            metric: value.numpy()
            for metric, value in ret_metrics.items()
        }
        if nan_to_num is not None:
            ret_metrics = OrderedDict({
                metric: np.nan_to_num(metric_value, nan=nan_to_num)
                for metric, metric_value in ret_metrics.items()
            })
        return ret_metrics


    # 这里去计算那个表格
    def evaluate(self, size: int) -> dict:
        """Evaluate the model performance of the whole dataset after processing
        all batches.

        Args:
            size (int): Length of the entire validation dataset. When batch
                size > 1, the dataloader may pad some data samples to make
                sure all ranks have the same length of dataset slice. The
                ``collect_results`` function will drop the padded data based on
                this size.

        Returns:
            dict: Evaluation metrics dict on the val dataset. The keys are the
            names of the metrics, and the values are corresponding results.
        """
        if len(self.results) == 0:
            print_log(
                f'{self.__class__.__name__} got empty `self.results`. Please '
                'ensure that the processed results are properly added into '
                '`self.results` in `process` method.',
                logger='current',
                level=logging.WARNING)

        if self.collect_device == 'cpu':
            L1_results = collect_results(
                self.L1_results,
                size,
                self.collect_device,
                tmpdir=self.collect_dir)
            L2_results = collect_results(
                self.L2_results,
                size,
                self.collect_device,
                tmpdir=self.collect_dir)
            results = collect_results(   # L3
                self.results,
                size,
                self.collect_device,
                tmpdir=self.collect_dir)
        else:
            L1_results = collect_results(self.L1_results, size, self.collect_device)
            L2_results = collect_results(self.L2_results, size, self.collect_device)
            results = collect_results(self.results, size, self.collect_device)

        if is_main_process():
            # cast all tensors in results list to cpu


            # 打印L1的结果
            self.dataset_meta['classes'] = L1_classes_name
            L1_results = _to_cpu(L1_results)
            L1_metrics = self.compute_metrics(L1_results)
            print_log("L1 level metrics: " + ', '.join(f"{k}: {v}" for k, v in L1_metrics.items()), logger='current')
            print('\n')

            # 打印L2的结果
            self.dataset_meta['classes'] = L2_classes_name
            L2_results = _to_cpu(L2_results)
            L2_metrics = self.compute_metrics(L2_results)
            print_log("L2 level metrics: " + ', '.join(f"{k}: {v}" for k, v in L2_metrics.items()), logger='current')
            print('\n')# 

            # 打印L3的结果  / L2 / L1 根据self.test_output_level参数指定，恢复self.dataset_meta['classes']
            if self.test_output_level == 'L3':
                self.dataset_meta['classes'] = L3_classes_name
            elif self.test_output_level == 'L2':
                self.dataset_meta['classes'] = L2_classes_name
            elif self.test_output_level == 'L1':
                self.dataset_meta['classes'] = L1_classes_name
            results = _to_cpu(results)
            _metrics = self.compute_metrics(results)  # type: ignore # 这里就输出了那个啥啦

            # Add prefix to metric names
            if self.prefix: # self.prefix=None
                _metrics = {
                    '/'.join((self.prefix, k)): v
                    for k, v in _metrics.items()
                }
            metrics = [_metrics]
        else:
            metrics = [None]  # type: ignore

        broadcast_object_list(metrics)

        # reset the results list
        self.results.clear()
        return metrics[0] #最后的那个，这里直接打印出来得了


def _to_cpu(data: Any) -> Any:
    """transfer all tensors and BaseDataElement to cpu."""
    if isinstance(data, (Tensor, BaseDataElement)):
        return data.to('cpu')
    elif isinstance(data, list):
        return [_to_cpu(d) for d in data]
    elif isinstance(data, tuple):
        return tuple(_to_cpu(d) for d in data)
    elif isinstance(data, dict):
        return {k: _to_cpu(v) for k, v in data.items()}
    else:
        return data
