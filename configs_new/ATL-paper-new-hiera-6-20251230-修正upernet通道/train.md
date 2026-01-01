```bash
PORT=12345 CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh  /data/AI-Tianlong/openmmlab/mmsegmentation/configs_new/ATL-paper-new-hiera-6-20251230-修正upernet通道/baseline-GF2/part2-baseline-GF2-swin-B-win7-224pr-upernet-baseline-640x640.py 2 


PORT=12346 CUDA_VISIBLE_DEVICES=2,3 bash tools/dist_train.sh /data/AI-Tianlong/openmmlab/mmsegmentation/configs_new/ATL-paper-new-hiera-6-20251230-修正upernet通道/BHCCM+LHSC-GF2/BHCCM+LHSC-GF2-swin-B-upernet-消融6.py 2 

PORT=12349 CUDA_VISIBLE_DEVICES=4,5 bash tools/dist_train.sh /data/AI-Tianlong/openmmlab/mmsegmentation/configs_new/ATL-paper-new-hiera-6-20251230-修正upernet通道/BHCCM+LHSC-GF2/BHCCM+LHSC-GF2-deeplabv3-消融6.py 2

```