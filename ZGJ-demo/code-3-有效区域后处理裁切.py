import argparse
import logging
import os
import tempfile

from osgeo import gdal, ogr, osr

gdal.DontUseExceptions()


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    return logging.getLogger(__name__)


def resolve_paths(image_path, vector_path_arg, output_path_arg):
    image_stem = os.path.splitext(os.path.basename(image_path))[0]

    # VECTOR_PATH: 传目录则自动拼接同名 geojson，传文件则直接使用。
    if os.path.isdir(vector_path_arg):
        vector_path = os.path.join(vector_path_arg, f'{image_stem}.geojson')
    else:
        vector_path = vector_path_arg

    # OUTPUT_PATH: 传目录则自动拼接同名 tif，传 .tif 文件则直接使用。
    output_is_dir = os.path.isdir(output_path_arg)
    if (not output_is_dir) and (not os.path.splitext(output_path_arg)[1]):
        output_is_dir = True

    if output_is_dir:
        output_path = os.path.join(output_path_arg, f'{image_stem}.tif')
    else:
        output_path = output_path_arg

    return vector_path, output_path


def clip_raster_by_vector(image_path, vector_path, output_path):
    src_ds = gdal.Open(image_path)
    if src_ds is None:
        raise RuntimeError(f"无法打开输入影像: {image_path}")

    raster_wkt = src_ds.GetProjectionRef()
    raster_srs = None
    if raster_wkt:
        raster_srs = osr.SpatialReference()
        raster_srs.ImportFromWkt(raster_wkt)
        raster_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    else:
        logging.warning('输入影像未读取到投影信息。')

    band1 = src_ds.GetRasterBand(1)
    src_nodata = band1.GetNoDataValue() if band1 is not None else None

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 如果输入矢量缺失SRS，先临时写出一个带影像SRS的矢量再用于cutline。
    cutline_for_warp = vector_path
    tmp_cutline = None
    vec_ds = ogr.Open(vector_path)
    if vec_ds is None:
        src_ds = None
        raise RuntimeError(f"无法打开输入矢量: {vector_path}")

    vec_layer = vec_ds.GetLayer(0)
    vec_srs = vec_layer.GetSpatialRef() if vec_layer is not None else None
    vec_has_srs = False
    if vec_srs is not None:
        try:
            vec_has_srs = bool(vec_srs.ExportToWkt())
        except Exception:
            vec_has_srs = False

    if vec_layer is None:
        vec_ds = None
        src_ds = None
        raise RuntimeError('输入矢量不包含可用图层。')

    logging.info(f'矢量要素数: {vec_layer.GetFeatureCount()}')

    if (not vec_has_srs) and raster_srs is not None:
        logging.warning('输入矢量缺少坐标系，自动使用输入影像坐标系进行裁切。')
        tmp_dir = tempfile.mkdtemp(prefix='cutline_fix_')
        tmp_cutline = os.path.join(tmp_dir, 'cutline_with_srs.gpkg')
        gpkg_drv = ogr.GetDriverByName('GPKG')
        out_vec_ds = gpkg_drv.CreateDataSource(tmp_cutline)
        out_lyr = out_vec_ds.CreateLayer('cutline', srs=raster_srs, geom_type=ogr.wkbUnknown)
        out_lyr.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))

        vec_layer.ResetReading()
        idx = 1
        for feat in vec_layer:
            geom = feat.GetGeometryRef()
            if geom is None:
                continue
            out_feat = ogr.Feature(out_lyr.GetLayerDefn())
            out_feat.SetField('id', idx)
            out_feat.SetGeometry(geom.Clone())
            out_lyr.CreateFeature(out_feat)
            out_feat = None
            idx += 1

        out_vec_ds = None
        cutline_for_warp = tmp_cutline

    vec_ds = None

    cutline_srs_wkt = raster_wkt if raster_wkt else None
    if cutline_srs_wkt is None and (not vec_has_srs):
        src_ds = None
        raise RuntimeError('影像和矢量都缺少坐标系，无法执行裁切。')

    warp_options = gdal.WarpOptions(
        format='GTiff',
        cutlineDSName=cutline_for_warp,
        cutlineSRS=cutline_srs_wkt,
        cropToCutline=True,
        dstNodata=src_nodata,
        srcNodata=src_nodata,
        multithread=True,
        resampleAlg='near',
        warpOptions=[
            'CUTLINE_ALL_TOUCHED=TRUE',
            'NUM_THREADS=ALL_CPUS',
        ],
    )

    out_ds = gdal.Warp(output_path, src_ds, options=warp_options)
    if out_ds is None:
        if tmp_cutline and os.path.exists(tmp_cutline):
            ogr.GetDriverByName('GPKG').DeleteDataSource(tmp_cutline)
            tmp_parent = os.path.dirname(tmp_cutline)
            if os.path.isdir(tmp_parent):
                os.rmdir(tmp_parent)
        src_ds = None
        raise RuntimeError('按矢量裁切失败。')

    out_ds.FlushCache()
    out_ds = None

    if tmp_cutline and os.path.exists(tmp_cutline):
        ogr.GetDriverByName('GPKG').DeleteDataSource(tmp_cutline)
        tmp_parent = os.path.dirname(tmp_cutline)
        if os.path.isdir(tmp_parent):
            os.rmdir(tmp_parent)

    src_ds = None


def build_arg_parser():
    parser = argparse.ArgumentParser(description='用矢量裁切单景影像')
    parser.add_argument('--IMAGE_PATH', required=True, help='输入影像路径')
    parser.add_argument('--VECTOR_PATH', required=True, help='输入矢量路径，支持文件或目录（目录时自动拼接同名 .geojson）')
    parser.add_argument('--OUTPUT_PATH', required=True, help='输出路径，支持 .tif 文件或目录（目录时自动拼接同名 .tif）')
    return parser


def main():
    print('\n')
    logger = setup_logger()
    args = build_arg_parser().parse_args()

    if not os.path.isfile(args.IMAGE_PATH):
        raise FileNotFoundError(f"输入影像不存在: {args.IMAGE_PATH}")

    vector_path, output_path = resolve_paths(args.IMAGE_PATH, args.VECTOR_PATH, args.OUTPUT_PATH)

    if not os.path.isfile(vector_path):
        raise FileNotFoundError(f"输入矢量不存在: {vector_path}")

    logger.info(f"实际矢量路径: {vector_path}")
    logger.info(f"实际输出路径: {output_path}")

    logger.info('开始执行矢量裁切...')
    clip_raster_by_vector(args.IMAGE_PATH, vector_path, output_path)
    logger.info(f"裁切完成: {output_path}")


if __name__ == '__main__':
    main()




