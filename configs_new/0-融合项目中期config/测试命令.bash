
# conda activate atl-py10
# 1 第1组数据 baseline
CUDA_VISIBLE_DEVICES=0,1,2,5 bash tools/dist_test.sh configs_new/0-融合项目中期config/1-第1组数据-part2-baseline-Google-swin-B-win7-224pr-upernet-baseline-896x896.py work_dirs/part2-baseline-Google-swin-B-win7-224pr-upernet-baseline-896x896/iter_80000.pth 4

# 1 第1组数据 BHCCM
CUDA_VISIBLE_DEVICES=0,1,2,5 bash tools/dist_test.sh configs_new/0-融合项目中期config/1-第1组数据-BHCCM+LHSC-Google-swin-B-upernet-消融6.py work_dirs/BHCCM+LHSC-Google-swin-B-upernet-继续训练/iter_72000.pth 4


# 2 第2组数据 baseline
CUDA_VISIBLE_DEVICES=0,1,2,5 bash tools/dist_test.sh configs_new/0-融合项目中期config/2-第2组数据-part2-baseline-Google-swin-B-win7-224pr-upernet-baseline-896x896.py work_dirs/part2-baseline-Google-swin-B-win7-224pr-upernet-baseline-896x896/iter_80000.pth 4

# 2 第2组数据 BHCCM
PORT=15321 CUDA_VISIBLE_DEVICES=0,1,2,5 bash tools/dist_test.sh configs_new/0-融合项目中期config/2-第2组数据-BHCCM+LHSC-Google-swin-B-upernet-消融6.py work_dirs/BHCCM+LHSC-Google-swin-B-upernet-继续训练/iter_72000.pth 4





# 3 第3组数据 baseline
CUDA_VISIBLE_DEVICES=0,1,2,5 bash tools/dist_test.sh configs_new/0-融合项目中期config/3-第3组数据-part2-baseline-Google-swin-B-win7-224pr-upernet-baseline-896x896.py work_dirs/part2-baseline-Google-swin-B-win7-224pr-upernet-baseline-896x896/iter_80000.pth 4

# 3 第3组数据 BHCCM
CUDA_VISIBLE_DEVICES=0,1,2,5 bash tools/dist_test.sh configs_new/0-融合项目中期config/3-第3组数据-BHCCM+LHSC-Google-swin-B-upernet-消融6.py work_dirs/BHCCM+LHSC-Google-swin-B-upernet-继续训练/iter_72000.pth 4


