
import os

from osgeo import gdal

from mmseg.apis import init_model, inference_model
from collections import defaultdict
from typing import Sequence, Union
import sys
from argparse import ArgumentParser
from tqdm import tqdm

import numpy as np
from mmengine.dataset import Compose
from mmengine.model import BaseModel

ImageType = Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]]

from mmengine import Config
from mmseg.models import BaseSegmentor
from mmseg.registry import MODELS
from mmengine.runner import load_checkpoint
from mmengine.utils import mkdir_or_exist
import torch

from mmseg.datasets.transforms.formatting import PackSegInputs

# from opencd.datasets.transforms.loading import MultiImgLoadImageFromFile_gdal
# from opencd.datasets.transforms.formatting import MultiImgPackSegInputs

from mmcv.transforms import to_tensor

# Ensure workspace root is on sys.path for src_utils imports
current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.abspath(os.path.join(current_dir, '../../'))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

# from src_utils.xml_parse.xml_parse import parse_xml_config
# from src_utils.logging.logging import setup_logger
from ATL_Tools import setup_logger


import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
import os
from osgeo import gdal

def process_single_explicit_slide(model,
                               t1_path, 
                               save_path_raw,
                               num_classes=18, # 需要知道类别数以建立缓存
                               window_size = 512,
                               stride = 384):
    """ 基于滑动窗口重叠平均逻辑的处理函数 """
    
    # 1. 打开数据
    ds1 = gdal.Open(t1_path)
    if ds1 is None:
        raise FileNotFoundError(f"Cannot open t1: {t1_path}")

    w, h = ds1.RasterXSize, ds1.RasterYSize
    bands = ds1.RasterCount
    
    logging.info(f"执行重叠滑窗推理: {os.path.basename(t1_path)} | 尺寸: {w}x{h}")

    # 2. 初始化累加器和计数器 (放在内存中)
    # 注意：如果图像极大且类别多，这里可能会爆内存。
    # 假设图像 10000x10000, 10类, float32, 约需 3.7GB RAM
    full_preds = np.zeros((num_classes, h, w), dtype=np.float32)
    count_mat = np.zeros((1, h, w), dtype=np.float32)

    # 3. 计算网格坐标
    h_grids = max(h - window_size + stride - 1, 0) // stride + 1
    w_grids = max(w - window_size + stride - 1, 0) // stride + 1

    # 4. 滑窗循环
    for h_idx in tqdm(range(h_grids), desc="Height Grids"):
        for w_idx in range(w_grids):
            # --- 核心逻辑：坐标回退 ---
            y1 = h_idx * stride
            x1 = w_idx * stride
            y2 = min(y1 + window_size, h)
            x2 = min(x1 + window_size, w)
            y1 = max(y2 - window_size, 0) # 如果触碰右/下边界，则向左/上回退
            x1 = max(x2 - window_size, 0)

            # 读取数据
            t1_np = ds1.ReadAsArray(x1, y1, window_size, window_size).transpose(1, 2, 0)

            # 如果全 0 则跳过处理（加速处理背景）
            if np.all(t1_np == 0):
                count_mat[:, y1:y2, x1:x2] += 1
                continue

            # 格式化数据
            # 这里的 transform 逻辑保持你的 PackSegInputs 即可
            results = {
                'img': t1_np,
                'img_shape': t1_np.shape[:2],
                'ori_shape': t1_np.shape[:2]
            }
            # 这里的 transform 需要你自己之前定义的 Compose([PackSegInputs()])
                    # 格式化数据 # 这个只用那个ImgPackSegInputs
            transform = Compose([
                PackSegInputs()
             ])
            packed_data = transform(results) 

            data = {
            'inputs': [packed_data['inputs']],
            'data_samples': [packed_data['data_samples']]
            }

            # ... (此处省略你原有的 data 组装过程)

            # 推理
            with torch.no_grad():
                # 注意：为了实现平均，我们需要模型的 Logits (概率分布)，而不是 Argmax 后的类别 ID
                # 在 MMSeg 中，DataSample 通常包含 seg_logits
                model_output = model.test_step(data) 
                # 获取该 Patch 的 Logits [C, H, W]
                crop_logits = model_output[0].seg_logits.data.cpu().numpy()

            # --- 核心逻辑：累加结果 ---
            full_preds[:, y1:y2, x1:x2] += crop_logits
            count_mat[:, y1:y2, x1:x2] += 1

    # 5. 计算平均值并取最大概率类别
    logging.info("正在聚合结果并生成最终 Mask...")
    final_logits = full_preds / count_mat

    reduce_zero_label = True
    if reduce_zero_label:
        final_mask = (np.argmax(final_logits, axis=0)+1).astype(np.uint8)
    else:
        final_mask = np.argmax(final_logits, axis=0).astype(np.uint8)

    # 6. 写入文件 (一次性写入)
    driver = gdal.GetDriverByName("GTiff")
    out_mask_ds = driver.Create(save_path_raw, w, h, 1, gdal.GDT_Byte, options=["COMPRESS=LZW"])
    out_mask_ds.SetGeoTransform(ds1.GetGeoTransform())
    out_mask_ds.SetProjection(ds1.GetProjection())
    out_mask_ds.GetRasterBand(1).WriteArray(final_mask)
    
    # 7. 清理
    out_mask_ds = None
    ds1 = None
    logging.info(f"保存完成: {save_path_raw}")

def process_single_explicit(model,
                            t1_path, 
                            save_path_raw,
                            window_size = 512,
                            stride = 384, 
                            ):
    """ 处理单对大图，指定输出路径 """
    
    # 1. 打开数据
    ds1 = gdal.Open(t1_path)

    if ds1 is None:
        raise FileNotFoundError(f"Cannot open t1: {t1_path}")

    w, h = ds1.RasterXSize, ds1.RasterYSize
    bands = ds1.RasterCount
    
    logging.info(f"执行哨兵二号地物分类: 影像: {os.path.basename(t1_path)} | 尺寸: {w}x{h}")
    logging.info(f"输出 Mask: {save_path_raw}")

    # 2. 创建输出文件流
    driver = gdal.GetDriverByName("GTiff")
    out_mask_ds = driver.Create(save_path_raw, w, h, 1, gdal.GDT_Byte, options=["COMPRESS=LZW"])
    out_mask_ds.SetGeoTransform(ds1.GetGeoTransform())
    out_mask_ds.SetProjection(ds1.GetProjection())
    
    # 3. 滑窗循环
    windows = []
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            windows.append((x, y))
    
    for x, y in tqdm(windows, desc=f"Infer {os.path.basename(save_path_raw)}", leave=False):
        w_curr = min(window_size, w - x)
        h_curr = min(window_size, h - y)


        t1_np = ds1.ReadAsArray(x, y, w_curr, h_curr).transpose(1,2,0)  # [H,W,C]

        # 如果两个时相都是全 0，直接跳过并写入全 0 结果
        if np.all(t1_np == 0):
            pred_valid = np.zeros((h_curr, w_curr), dtype=np.uint8)
            out_mask_ds.GetRasterBand(1).WriteArray(pred_valid, x, y)
            continue
        
        # Padding (边缘处理)
        if w_curr < window_size or h_curr < window_size:
            t1_pad = np.zeros((window_size, window_size, bands), dtype=t1_np.dtype)
            t1_pad[:h_curr, :w_curr, :] = t1_np
            
            t1_in = t1_pad
        else:
            t1_in = t1_np


        # 格式化数据 # 这个只用那个ImgPackSegInputs
        transform = Compose([
            # MultiImgPackSegInputs()
            PackSegInputs()
        ])


        results=dict()
        results['img']=t1_in  # [512,512,4], [512,512,4]
        results['img_shape'] = t1_in.shape[:2]
        results['ori_shape'] = t1_in.shape[:2]

        packed_data = transform(results)
        
        data = {
            'inputs': [packed_data['inputs']],
            'data_samples': [packed_data['data_samples']]
        }

        with torch.no_grad():
            results = model.test_step(data)
        
        pred = results[0].pred_sem_seg.cpu().data.numpy()[0]

                # 裁剪有效区域
        pred_valid = pred[:h_curr, :w_curr]+1 # reduce_zero_label 需要加回来

        # 写入 Mask 文件 (存 0,1,2)
        out_mask_ds.GetRasterBand(1).WriteArray(pred_valid, x, y)

    # 4. 清理资源
    out_mask_ds = None
    ds1 = None
    ds2 = None
    logging.info(f"大图已推理完成！ 输出保存至: {save_path_raw}") 


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--config_path', help='Config file', default='/opt/workspace/AI-Tianlong/ZGJ/0-检察遥感平台部署/1-哨兵2号-地物分类程序/X-code/mmsegmentation/configs_new/ZGJ/2-18类地物/ZGJ-S2-BEiTv2-mask2former-18类-512-new.py')
    parser.add_argument('--checkpoint_path', help='Checkpoint file', default='/opt/workspace/AI-Tianlong/ZGJ/0-检察遥感平台部署/1-哨兵2号-地物分类程序/X-code/mmsegmentation/checkpoints/ZGJ-S2-BEiTv2-mask2former-18类-512-2-iter_80000.pth')

    parser.add_argument('--window-size', default=(512, 512), type=int, nargs=2,
                        help='window xsize,ysize')
    parser.add_argument('--stride', default=(384, 384), type=int, nargs=2,
                        help='window xstride,ystride')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')


    parser.add_argument("--DATA_INPUT_DIR1", type=str, 
                        default='/opt/workspace/AI-Tianlong/ZGJ/0-检察遥感平台部署/1-哨兵2号-地物分类程序/0-测试数据/2-波段组合(裁剪-拼接)结果/S2C_MSIL2A_20260309T025531_N0512_R032_T50SMJ_20260309T064312.tif',
                        help="哨兵二号影像路径")
    parser.add_argument("--DATA_INPUT_DIR2", type=str, 
                        default=None, 
                        help="可选的矢量范围路径")
    parser.add_argument("--DATA_OUTPUT_DIR", type=str, 
                        default='/opt/workspace/AI-Tianlong/ZGJ/0-检察遥感平台部署/1-哨兵2号-地物分类程序/0-测试数据/x-推理结果-大图推理/',
                        help="输出目录")

    args = parser.parse_args()

    logging = setup_logger()
    logging.info("准备执行基于mmseg模型的哨兵二号影像地物分类……")

    config = Config.fromfile(args.config_path)
    model = MODELS.build(config.model)

    checkpoint = torch.load(args.checkpoint_path, map_location='cpu', weights_only=False)

    model.load_state_dict(checkpoint['state_dict'])

    model.to('cuda:0')
    model.eval()

    # 使用封装好的函数解析 XML

    img1_path = os.path.join(args.DATA_INPUT_DIR1)
    img_basename = os.path.basename(args.DATA_INPUT_DIR1)
    output_semantic_mask_path = os.path.join(args.DATA_OUTPUT_DIR, img_basename) # 固定的名字
    output_dir = args.DATA_OUTPUT_DIR
    mkdir_or_exist(output_dir)
    
    try:
        # process_single_explicit(model, 
        process_single_explicit_slide(model, 
                                img1_path, 
                                output_semantic_mask_path,
                                window_size=512,
                                stride=384)

    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
            
    logging.info(f"step4：mmseg 模型推理已完成！")
