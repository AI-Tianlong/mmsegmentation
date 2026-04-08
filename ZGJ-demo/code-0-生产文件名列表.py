from ATL_Tools import find_data_list, mkdir_or_exist
import os 

img_list = find_data_list('/opt/workspace/AI-Tianlong/ZGJ/0-检察遥感平台部署/1-哨兵2号-地物分类程序/0-测试数据/2-波段组合结果', suffix='.tif')

with open('/opt/workspace/AI-Tianlong/ZGJ/0-检察遥感平台部署/1-哨兵2号-地物分类程序/X-code/img_txt.txt', 'w+') as f:
    for img_name in img_list:
        f.write(os.path.basename(img_name)+'\n')
