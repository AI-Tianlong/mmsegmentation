import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from ATL_Tools import mkdir_or_exist, find_data_list
from ATL_Tools.ATL_gdal import crop_tif_with_json_nan, Mosaic_all_imgs
import argparse
import os
from osgeo import gdal
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert mask to RGB image with geoinfo')
    parser.add_argument('mask_label_path', help='mask folder path')
    parser.add_argument('rgb_output_path', help='rgb output folder path')
    parser.add_argument('--sensor', default='S2',help='GF2 or S2')
    parser.add_argument('--name', default='5B-L3', help='5B-L3')
    parser.add_argument('--reduce', default=True, help='reduce_zero_label')
    parser.add_argument('--backend', default='gdal', help='gdal or PIL')

    args = parser.parse_args()
    return args


def mask2RGB_with_Geoinfo(
        MASK_path: str,
        RGB_path: str,
        IMG_path: str,
        backend='gdal'):
    
    """把mask转为RGB图像，并添加坐标信息

    Args:
        MASK_path (str): mask路径
        RGB_path (str): RGB图像路径
        IMG_path (str): 原始图像路径

    Returns:
        None, 生成的RGB图像保存在RGB_path
    """

    mkdir_or_exist(RGB_path)

    label_suffix = '.png'

    # 是否包含0类，classes和palette里没包含
    reduce_zero_label = args.reduce
    # 给生成的RGB图像添加坐标
    add_meta_info = True 

    label_lists = find_data_list(MASK_path, suffix=label_suffix)



    palette = METAINFO['palette']


    if reduce_zero_label:
        new_palette = [[0, 0, 0]] + palette
        print(f"palette: {new_palette}")
    else:
        new_palette = palette
        print(f"palette: {new_palette}")

    new_palette = np.array(new_palette)


    for mask_label_path in tqdm(label_lists, colour='Green'):
        mask_label = np.array(Image.open(mask_label_path)).astype(np.uint8)
        mask_label[mask_label == 255] = 0
        h,w = mask_label.shape

        RGB_label = new_palette[mask_label].astype(np.uint8)
        

        if backend == 'PIL':
            output_path = os.path.join(RGB_path, os.path.basename(mask_label_path).replace(label_suffix, '.png'))
            RGB_label = Image.fromarray(RGB_label).save(output_path)
        elif backend == 'gdal':
            output_path = os.path.join(RGB_path, os.path.basename(mask_label_path).replace(label_suffix, '.tif'))
            driver = gdal.GetDriverByName('GTiff')
            RGB_label_gdal = driver.Create(output_path, w, h, 3, gdal.GDT_Byte)

            RGB_label_gdal.GetRasterBand(1).WriteArray(RGB_label[:,:,0])
            RGB_label_gdal.GetRasterBand(2).WriteArray(RGB_label[:,:,1])
            RGB_label_gdal.GetRasterBand(3).WriteArray(RGB_label[:,:,2])

            if add_meta_info:
                IMG_file_path = os.path.join(IMG_path, os.path.basename(mask_label_path).replace(label_suffix, '.tif'))
                IMG_gdal = gdal.Open(IMG_file_path, gdal.GA_ReadOnly)
                assert  IMG_gdal is not None, f"无法打开 {os.path.join(IMG_path, os.path.basename(mask_label_path).replace(label_suffix, '.tif'))}"

                trans = IMG_gdal.GetGeoTransform()
                proj = IMG_gdal.GetProjection()

                RGB_label_gdal.SetGeoTransform(trans)
                RGB_label_gdal.SetProjection(proj)

            RGB_label_gdal = None

if __name__ == '__main__':
    args = parse_args()
    
    if args.sensor == 'GF2' and args.name == '5B-L3' or args.name == '5B-L2' or args.name == '5B-L1':
        img_folder = '/data/AI-Tianlong/openmmlab/mmsegmentation/data/1-paper-segmentation/2-多领域地物覆盖基础/0-GF2-5B-18-640/img_dir/val'
    elif args.sensor == 'S2' and args.name == '5B-L3' or args.name == '5B-L2' or args.name == '5B-L1':
        img_folder = '/data/AI-Tianlong/openmmlab/mmsegmentation/data/1-paper-segmentation/2-多领域地物覆盖基础/0-S2-5B-18-512/img_dir/val'
    elif args.name == 'crop10m':
        img_folder = '/data/AI-Tianlong/openmmlab/mmsegmentation/data/1-paper-segmentation/2-多领域地物覆盖-S2-crop作物/img_dir/val'


    if args.name == '5B-L3':
        METAINFO = dict(
            classes=('Paddy field', 'Other Field', 'Forest', 'Natural meadow',
                    'Artificial meadow', 'River', 'Lake', 'Pond',
                    'Factory-Storage-Shopping malls', 'Urban residential',
                    'Rural residential', 'Stadium', 'Park Square', 'Road',
                    'Overpass', 'Railway station', 'Airport', 'Bare land''Glaciers Snow'),
            palette=[[0,   240, 150], [150, 250, 0  ], [0,   150, 0  ], [250, 200, 0  ],
                    [200, 200, 0  ], [0,   0,   200], [0,   150, 200], [150, 200, 250],
                    [200, 0,   0  ], [250, 0,   150], [200, 150, 150], [250, 200, 150],
                    [150, 150, 0  ], [250, 150, 150], [250, 150, 0  ], [250, 200, 250],
                    [200, 150, 0  ], [200, 100, 50 ]])                              
    elif args.name == '5B-L2':
        METAINFO = dict(
            classes=('耕地','林地','草地','水体','仓储工矿商业地，大面积的那种',
                    '住宅用地','公共设施','交通设施','裸地'),
            palette=[[112, 236, 89], [0, 150, 0], [250, 200, 0], [0, 100, 255],
                    [200, 0, 0],[255, 217, 102],[250, 200, 150], [250, 150, 0],[198, 89, 17]])        
    elif args.name == '5B-L1':
        METAINFO = dict(
            classes=('植被','水体','人造地表','裸地'),
            palette=[[146, 208, 80], [0, 100, 255], [255, 217, 102], [198, 89, 17]])                              
    elif args.name == 'crop10m':
        METAINFO = dict(
            classes=('Rice', 'Corn', 'soybean', 'Not-Farmland'),
            palette=[[0, 200, 0], [250, 200, 0], [250, 0, 150], [255, 255, 255]])

  
    mask2RGB_with_Geoinfo(MASK_path=args.mask_label_path, 
                          RGB_path=args.rgb_output_path,
                          IMG_path=img_folder, 
                          backend=args.backend)
