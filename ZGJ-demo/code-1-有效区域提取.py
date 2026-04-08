import os


from osgeo import gdal, ogr, osr

import shutil
gdal.DontUseExceptions()
import argparse
import numpy as np

from ATL_Tools import mkdir_or_exist, find_data_list, setup_logger


def collect_raster_files(input_dir):
    """递归收集常见遥感影像文件。"""
    valid_ext = {'.tif', '.tiff', '.img', '.jp2'}
    raster_files = []
    for root, _, files in os.walk(input_dir):
        for name in files:
            if os.path.splitext(name)[1].lower() in valid_ext:
                raster_files.append(os.path.join(root, name))
    return sorted(raster_files)


def extract_sentinel_valid_regions_to_geojson(input_dir,
                                              output_geojson,
                                              nodata_value=0):
    """批量提取目录中影像的有效区域，并保存到 GeoJSON。"""
    raster_files = collect_raster_files(input_dir)
    if len(raster_files) == 0:
        logging.error(f"未在目录中找到影像文件: {input_dir}")
        return False

    logging.info(f"待处理影像数量: {len(raster_files)}")
    mkdir_or_exist(os.path.dirname(output_geojson) if os.path.dirname(output_geojson) else '.')

    drv = ogr.GetDriverByName('GeoJSON')
    if os.path.exists(output_geojson):
        drv.DeleteDataSource(output_geojson)
    ds_out = drv.CreateDataSource(output_geojson)

    target_srs = None
    layer = ds_out.CreateLayer(
        'valid_regions',
        srs=target_srs,
        geom_type=ogr.wkbMultiPolygon,
        options=['RFC7946=NO'])
    layer.CreateField(ogr.FieldDefn('image', ogr.OFTString))
    layer.CreateField(ogr.FieldDefn('status', ogr.OFTString))

    merged_geom = None
    success_count = 0
    skip_count = 0

    for idx, img_path in enumerate(raster_files, start=1):
        basename = os.path.basename(img_path)
        logging.info(f"[{idx}/{len(raster_files)}] 正在处理: {basename}")

        ds = gdal.Open(img_path)
        if ds is None:
            logging.warning(f"无法打开影像，跳过: {img_path}")
            skip_count += 1
            continue

        geom = get_valid_footprint(ds, basename=basename, nodata_value=nodata_value)
        if geom is None or geom.IsEmpty():
            logging.warning(f"未提取到有效区域，跳过: {basename}")
            ds = None
            skip_count += 1
            continue

        src_srs = osr.SpatialReference()
        src_srs.ImportFromWkt(ds.GetProjectionRef())
        src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        if target_srs is None and src_srs is not None:
            target_srs = src_srs.Clone()
            # 重新创建图层，确保图层SRS和影像一致
            ds_out = None
            if os.path.exists(output_geojson):
                drv.DeleteDataSource(output_geojson)
            ds_out = drv.CreateDataSource(output_geojson)
            layer = ds_out.CreateLayer(
                'valid_regions',
                srs=target_srs,
                geom_type=ogr.wkbMultiPolygon,
                options=['RFC7946=NO'])
            layer.CreateField(ogr.FieldDefn('image', ogr.OFTString))
            layer.CreateField(ogr.FieldDefn('status', ogr.OFTString))

        geom_out = geom.Clone()
        if target_srs and src_srs and not src_srs.IsSame(target_srs):
            transformer = osr.CoordinateTransformation(src_srs, target_srs)
            geom_out.Transform(transformer)

        geom_out = geom_out.Buffer(0)
        if geom_out is None or geom_out.IsEmpty():
            ds = None
            skip_count += 1
            continue

        if geom_out.GetGeometryType() == ogr.wkbPolygon:
            mp = ogr.Geometry(ogr.wkbMultiPolygon)
            mp.AddGeometry(geom_out)
            geom_out = mp

        feat = ogr.Feature(layer.GetLayerDefn())
        feat.SetField('image', basename)
        feat.SetField('status', 'single')
        feat.SetGeometry(geom_out)
        layer.CreateFeature(feat)
        feat = None

        if merged_geom is None:
            merged_geom = geom_out.Clone()
        else:
            merged_geom = merged_geom.Union(geom_out)

        ds = None
        success_count += 1

    if merged_geom is not None and (not merged_geom.IsEmpty()):
        merged_geom = merged_geom.Buffer(0)
        if merged_geom.GetGeometryType() == ogr.wkbPolygon:
            mp = ogr.Geometry(ogr.wkbMultiPolygon)
            mp.AddGeometry(merged_geom)
            merged_geom = mp

        feat_all = ogr.Feature(layer.GetLayerDefn())
        feat_all.SetField('image', '__ALL__')
        feat_all.SetField('status', 'merged')
        feat_all.SetGeometry(merged_geom)
        layer.CreateFeature(feat_all)
        feat_all = None

    ds_out = None

    logging.info(f"GeoJSON 保存完成: {output_geojson}")
    if target_srs is not None:
        logging.info(f"输出图层SRS(WKT): {target_srs.ExportToWkt()[:300]}...")
    logging.info(f"成功提取: {success_count} 景, 跳过: {skip_count} 景")
    return success_count > 0


def extract_single_image_valid_region_to_geojson(image_path,
                                                 output_geojson,
                                                 nodata_value=0):
    """提取单景影像有效区域并保存到 GeoJSON。"""
    if not os.path.exists(image_path):
        logging.error(f"影像不存在: {image_path}")
        return False

    ds = gdal.Open(image_path)
    if ds is None:
        logging.error(f"无法打开影像: {image_path}")
        return False

    geom = get_valid_footprint(
        ds,
        basename=os.path.basename(image_path),
        nodata_value=nodata_value,
        use_all_bands=True)
    if geom is None or geom.IsEmpty():
        logging.error("未提取到有效区域。")
        ds = None
        return False

    proj_wkt = ds.GetProjectionRef()
    src_srs = None
    if proj_wkt:
        src_srs = osr.SpatialReference()
        src_srs.ImportFromWkt(proj_wkt)
        src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    # 保持与输入影像一致的投影，不做重投影。
    geom_out = geom.Clone()

    geom_out = geom_out.Buffer(0)
    if geom_out.GetGeometryType() == ogr.wkbPolygon:
        mp = ogr.Geometry(ogr.wkbMultiPolygon)
        mp.AddGeometry(geom_out)
        geom_out = mp

    mkdir_or_exist(os.path.dirname(output_geojson) if os.path.dirname(output_geojson) else '.')
    drv = ogr.GetDriverByName('GeoJSON')
    if os.path.exists(output_geojson):
        drv.DeleteDataSource(output_geojson)
    ds_out = drv.CreateDataSource(output_geojson)

    layer = ds_out.CreateLayer(
        'valid_region',
        srs=src_srs,
        geom_type=ogr.wkbMultiPolygon,
        options=['RFC7946=NO'])
    layer.CreateField(ogr.FieldDefn('image', ogr.OFTString))
    layer.CreateField(ogr.FieldDefn('status', ogr.OFTString))

    feat = ogr.Feature(layer.GetLayerDefn())
    feat.SetField('image', os.path.basename(image_path))
    feat.SetField('status', 'single')
    feat.SetGeometry(geom_out)
    layer.CreateFeature(feat)

    feat = None
    ds_out = None
    ds = None
    logging.info(f"单景有效区域 GeoJSON 已保存: {output_geojson}")
    if src_srs is not None:
        logging.info(f"输出图层SRS(WKT): {src_srs.ExportToWkt()[:300]}...")
    return True


def get_valid_footprint(ds, 
                        basename='',
                        nodata_value=0,
                        use_all_bands=False):
    """
    读取栅格图像，提取有效区域（非 NoData）的矢量多边形。
    使用 gdal.Polygonize 实现，比 OpenCV 降采样更精确。
    """
    logging.info(f"提取影像 {basename} 的有效区域...")
    
    # 2. 默认读取第一个波段；可选按所有波段联合判定有效区
    band = ds.GetRasterBand(1)
    
    # 获取或设置 NoData 值
    meta_nodata = band.GetNoDataValue()
    if meta_nodata is None:
        meta_nodata = nodata_value
        
    # 创建二值掩膜 (Mask): 有效区域=1, 无效区域=0
    # 注意：这里使用内存中的虚拟栅格驱动来存储掩膜
    mask_ds = gdal.GetDriverByName('MEM').Create('', ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Byte)
    mask_ds.SetGeoTransform(ds.GetGeoTransform())
    mask_ds.SetProjection(ds.GetProjection())
    mask_band = mask_ds.GetRasterBand(1)
    
    # 分块处理以节省内存
    block_size = 2048
    width = ds.RasterXSize
    height = ds.RasterYSize
    
    for y in range(0, height, block_size):
        y_size = min(block_size, height - y)
        for x in range(0, width, block_size):
            x_size = min(block_size, width - x)
            
            if use_all_bands and ds.RasterCount > 1:
                band_masks = []
                for b_idx in range(1, ds.RasterCount + 1):
                    b = ds.GetRasterBand(b_idx)
                    b_data = b.ReadAsArray(x, y, x_size, y_size)
                    b_nodata = b.GetNoDataValue()
                    if b_nodata is None:
                        b_nodata = nodata_value
                    if b_nodata is not None:
                        # NaN 视为无效，避免边缘出现离散毛刺像素。
                        b_mask = np.isfinite(b_data) & (b_data != b_nodata)
                    else:
                        b_mask = np.isfinite(b_data)
                    band_masks.append(b_mask)
                # 多波段联合时采用“全部波段有效”更稳健，避免边缘被单波段噪声拉毛。
                valid_mask = np.logical_and.reduce(band_masks)
                mask_chunk = valid_mask.astype(np.uint8)
            else:
                data = band.ReadAsArray(x, y, x_size, y_size)
                if meta_nodata is not None:
                    mask_chunk = np.where(data != meta_nodata, 1, 0).astype(np.uint8)
                else:
                    mask_chunk = np.ones_like(data, dtype=np.uint8)
                
            mask_band.WriteArray(mask_chunk, x, y)
            
    mask_band.FlushCache()

    # 去除小碎斑，减少 Polygonize 后的锯齿和毛刺。
    gdal.SieveFilter(mask_band, None, mask_band, 64, 8)
    
    # 4. 栅格转矢量 (Polygonize)
    # 创建内存中的矢量图层
    ogr_ds = ogr.GetDriverByName('Memory').CreateDataSource('out')
    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjectionRef())
    layer = ogr_ds.CreateLayer('polygon', srs=srs)
    field_defn = ogr.FieldDefn('val', ogr.OFTInteger)
    layer.CreateField(field_defn)
    
    # 执行多边形化: 只提取像素值为 1 的区域
    # pixelValueField=0 表示将像素值写入第0个字段
    gdal.Polygonize(mask_band, mask_band, layer, 0, ['8CONNECTED=8'], callback=None)
    
    # 5. 合并多边形
    # Polygonize 可能会产生很多碎小的多边形，我们需要把所有 val=1 的几何体合并
    union_geom = None
    
    # 收集几何体
    count = layer.GetFeatureCount()
    if count == 0:
        return None
        
    for feature in layer:
        if feature.GetField(0) == 1:
            geom = feature.GetGeometryRef()
            if union_geom is None:
                union_geom = geom.Clone()
            else:
                union_geom = union_geom.Union(geom)
    
    # 清理
    if union_geom:
        union_geom = union_geom.Buffer(0) 
        
    return union_geom


# 定义函数：计算两个矩形的交集
def intersect(img1_ds, 
              img2_ds, 
              nodata_value=0, 
              coast_shp=None,
              img1_name='img1',
              img2_name='img2'):
    
    # 获取两个图像的有效多边形
    # 使用新函数 get_valid_footprint
    poly1 = get_valid_footprint(img1_ds, img1_name, nodata_value)
    poly2 = get_valid_footprint(img2_ds, img2_name, nodata_value)
    
    logging.info(f'两期影像有效区域提取成功，准备执行重采样与像素对齐操作……')
    if poly1 is None or poly2 is None:
        logging.warning("Warning: One of the images has no valid data.")
        return None
    
    # --- 统一坐标系 ---
    # 获取目标投影 (以 img1 为准)
    target_srs = osr.SpatialReference()
    target_srs.ImportFromWkt(img1_ds.GetProjectionRef())
    target_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    
    # img2 的 SRS
    source_srs2 = osr.SpatialReference()
    source_srs2.ImportFromWkt(img2_ds.GetProjectionRef())
    source_srs2.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    
    # 检查 poly2 的 SRS 是否与 img1 一致，不一致则转换
    if not target_srs.IsSame(source_srs2):
        logging.info("Reprojecting img2 footprint to match img1...")
        transform = osr.CoordinateTransformation(source_srs2, target_srs)
        poly2.Transform(transform)
        
    # 计算交集
    intersection = poly1.Intersection(poly2)
    
    if intersection is None or intersection.IsEmpty():
        return None

    # --- 处理海岸线矢量 (如果存在) ---
    if coast_shp and os.path.exists(coast_shp):
        logging.info(f"正在结合用户提供的海岸带矢量计算交集区域: {coast_shp}")
        
        ds_coast = ogr.Open(coast_shp)
        if not ds_coast:
            logging.warning(f"无法打开海岸线矢量: {coast_shp}")
        else:
            layer_coast = ds_coast.GetLayer()
            
            coast_srs = layer_coast.GetSpatialRef()
            if coast_srs:
                coast_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            
            transform_coast = None
            if coast_srs and not target_srs.IsSame(coast_srs):
                 logging.info("Reprojecting coast shapefile to match img1...")
                 transform_coast = osr.CoordinateTransformation(coast_srs, target_srs)
            
            # 使用 Union 合并矢量中的所有几何
            coast_union = None
            
            for feat in layer_coast:
                geom = feat.GetGeometryRef()
                if not geom: continue
                
                geom_clone = geom.Clone()
                if transform_coast:
                    geom_clone.Transform(transform_coast)
                
                # 优化: 只有当它和当前的 intersection 有交集时才合并
                if geom_clone.Intersects(intersection):
                    if coast_union is None:
                        coast_union = geom_clone
                    else:
                        coast_union = coast_union.Union(geom_clone)
            
            ds_coast = None
            
            if coast_union:
                # 修正可能出现的无效几何
                if not coast_union.IsValid():
                    coast_union = coast_union.MakeValid()
                    
                intersection = intersection.Intersection(coast_union)
                if intersection is None or intersection.IsEmpty():
                     logging.warning("与海岸线矢量求交集后结果为空！")
                     return None
            else:
                 logging.warning("没有与当前影像交集重叠的海岸线要素！")
                 return None
        
    final_env = intersection.GetEnvelope()
    # (minX, maxX, minY, maxY) (OpenMMLab 风格)
    
    # 重点：同时返回 边界(Bounds) 和 几何体WKT(Geometry WKT)
    return [final_env[0], final_env[2], final_env[1], final_env[3]], intersection.ExportToWkt()


def align_image(src_nodata_value,
                dst_nodata_value,
                img1_path:str,
                img2_path:str,
                output_dir: str,
                img_resampleAlg=gdal.GRA_Bilinear,
                coast_shp:str='',
                ):

    ref_ds = gdal.Open(img1_path)
    if ref_ds is None:
        logging.error(f"无法打开影像1: {img1_path}")
        return

    img1_ds = ref_ds

    img2_ds = gdal.Open(img2_path)
    if img2_ds is None:
        logging.error(f"无法打开影像2: {img2_path}")
        return

    # 检查并统一投影
    ref_proj = ref_ds.GetProjection()
    res = intersect(img1_ds, 
                    img2_ds, 
                    src_nodata_value, 
                    coast_shp=coast_shp,
                    img1_name=os.path.basename(img1_path),
                    img2_name=os.path.basename(img2_path))  # 计算源影像和目标影像的交集范围
    
    if res is None:
        logging.error(f"无有效交集区域, 程序退出")
        return
        
    bound, cutline_wkt = res
    logging.info(f'两期影像有效交集范围 bound: {bound}')

    # 创建矢量文件作为 Cutline (裁剪线)，保存到 output_dir 以便后续使用
    cutline_ds_name = os.path.join(output_dir, 'common_intersection_roi.shp')
    drv = ogr.GetDriverByName('ESRI Shapefile')
    
    # 如果已存在则删除，防止报错
    if os.path.exists(cutline_ds_name):
        drv.DeleteDataSource(cutline_ds_name)
        
    ds_cut = drv.CreateDataSource(cutline_ds_name)
    
    
    # 显式设置 SRS
    srs = osr.SpatialReference()
    srs.ImportFromWkt(ref_proj)
    
    layer_cut = ds_cut.CreateLayer('cutline', srs=srs, geom_type=ogr.wkbPolygon) 
    feat_cut = ogr.Feature(layer_cut.GetLayerDefn())
    geom_cut = ogr.CreateGeometryFromWkt(cutline_wkt)
    feat_cut.SetGeometry(geom_cut)
    layer_cut.CreateFeature(feat_cut)
    feat_cut = None
    ds_cut = None # 关闭保存

    # 设定输出分辨率为 2.0米
    resx = 2.0
    resy = -2.0 

    # --- 智能投影处理 ---
    dst_srs_wkt = ref_proj if ref_proj else None # 确保不是空字符串
    target_bound = bound

    if srs.IsGeographic():
        logging.info("检测到输入影像为地理坐标系 (WGS84等)，正在自动计算合适的 UTM 投影以支持 2m 分辨率...")
        # 计算中心经度
        idx_lon = (bound[0] + bound[2]) / 2  # bound 是 [minx, miny, maxx, maxy]
        idx_lat = (bound[1] + bound[3]) / 2
        
        # 计算 UTM 带号
        utm_band = int((idx_lon + 180) / 6) + 1
        epsg_code = 32600 + utm_band if idx_lat >= 0 else 32700 + utm_band
        
        logging.info("使用 WGS84+UTM 方式构建投影（不依赖 EPSG 数据库）。")
        dst_srs_obj = osr.SpatialReference()
        dst_srs_obj.SetWellKnownGeogCS("WGS84")
        dst_srs_obj.SetUTM(utm_band, idx_lat >= 0)
        dst_srs_wkt = dst_srs_obj.ExportToWkt()
        # 显式转换为 WKT 格式，gdal.Warp 有时对 osr对象支持不好，但对 WKT 字符串支持完美
        
        logging.info(f"已自动选择投影 EPSG:{epsg_code}")
        
        # 注意：当投影改变时，原有的 WGS84 bound 不再适用。
        # 设置为 None，让 gdal.Warp 根据 cutline 自动计算新的边界
        target_bound = None 

    gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')  # 设置GDAL使用所有CPU线程

    # 将重采样算法转换为字符串，避免 Swig 类型映射错误
    resample_alg_str = 'bilinear'
    if img_resampleAlg == gdal.GRA_NearestNeighbour: resample_alg_str = 'near'
    elif img_resampleAlg == gdal.GRA_Bilinear: resample_alg_str = 'bilinear'
    elif img_resampleAlg == gdal.GRA_Cubic: resample_alg_str = 'cubic'
    elif img_resampleAlg == gdal.GRA_CubicSpline: resample_alg_str = 'cubicspline'
    elif img_resampleAlg == gdal.GRA_Lanczos: resample_alg_str = 'lanczos'

    # 尝试使用关键字参数直接调用 gdal.Warp，避免手动构造 WarpOptions 对象
    # 这通常能避免 SWIG 类型映射错误
    warp_kwargs = {
        'format': 'GTiff',
        'xRes': resx, 
        'yRes': resy,
        'srcNodata': src_nodata_value, 
        'dstNodata': dst_nodata_value, 
        'srcAlpha': False,
        'resampleAlg': resample_alg_str,
        'cutlineDSName': cutline_ds_name,
        'cropToCutline': True,
        'dstSRS': dst_srs_wkt,    # 确保此处是 WKT 字符串
        'multithread': True
    }
    
    if target_bound is not None:
        warp_kwargs['outputBounds'] = target_bound

    img1_out_path = os.path.join(output_dir, os.path.basename(img1_path))
    img2_out_path = os.path.join(output_dir, os.path.basename(img2_path))

    logging.info(f"正在处理: {img1_out_path}")
    # 直接展开 kwargs
    img1_align = gdal.Warp(img1_out_path, img1_ds, **warp_kwargs)  
    
    logging.info(f"正在处理: {img2_out_path}")
    img2_align = gdal.Warp(img2_out_path, img2_ds, **warp_kwargs) 
 

    # 提示矢量已保存
    logging.info(f"交集区域矢量已保存至: {cutline_ds_name}")

    if img1_align and img2_align:
        logging.info(f'已完成交集提取与对齐:\n  {os.path.basename(img1_out_path)}\n  {os.path.basename(img2_out_path)}')

    # 显式关闭数据集以确保数据写入磁盘并释放资源
    img1_align = None
    img2_align = None
    img1_ds = None
    img2_ds = None

if __name__ == "__main__":

    logging = setup_logger()

    print('\n')
    logging.info("Step2：准备执行单景影像有效区域提取流程……")

    # 1. 参数解析
    parser = argparse.ArgumentParser()

    parser.add_argument("--IMAGE_PATH", type=str, 
                        help="单景影像绝对路径",
                        default='/opt/workspace/AI-Tianlong/ZGJ/0-检察遥感平台部署/1-哨兵2号-地物分类程序/0-测试数据/2-波段组合(裁剪-拼接)结果/S2C_MSIL2A_20260309T025531_N0512_R032_T50SMH_20260309T064312.tif')
    parser.add_argument("--OUTPUT_DIR", type=str,
                        help="GeoJSON输出文件夹路径",
                        default='/opt/workspace/AI-Tianlong/ZGJ/0-检察遥感平台部署/1-哨兵2号-地物分类程序/0-测试数据/3-有效区域提取')
    parser.add_argument("--NODATA_VALUE", type=float, required=False, default=0,
                        help="NoData像素值，默认0")

    args = parser.parse_args()

    image_base = os.path.splitext(os.path.basename(args.IMAGE_PATH))[0]
    output_geojson = os.path.join(args.OUTPUT_DIR, f"{image_base}.geojson")

    ok = extract_single_image_valid_region_to_geojson(
        image_path=args.IMAGE_PATH,
        output_geojson=output_geojson,
        nodata_value=args.NODATA_VALUE)
    if not ok:
        raise RuntimeError("单景哨兵有效区域提取失败，请检查日志。")

    logging.info(f"输出文件: {output_geojson}")
    logging.info("单景哨兵有效区域提取流程完成。")



