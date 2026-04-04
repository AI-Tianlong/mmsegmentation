# --------------------------------------------------------
# PIIP
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
from mmengine.config import Config, DictAction

from mmseg.models.backbones.piip_2branch import PIIPTwoBranch
from mmseg.models.backbones.piip_3branch import PIIPThreeBranch
# from piip_4branch import PIIPFourBranch

from mmseg.models.backbones.deit import vit_models
from mmseg.models.backbones.internvit_6b import InternViT6B


def convert_mmdet_config(cfg, config_name):
    # for flops calculation
    # cfg.pop("pretrained", None)
    cfg.pop("start_level", None)
    cfg.pop("_delete_", None)
    branch_pop_keys = ["with_fpn", "img_norm_cfg", "layerscale_force_fp32"]
    for branch in ["branch1", "branch2", "branch3", "branch4"]:
        # import pdb; pdb.set_trace()
        if branch in cfg:
            if 'beit' in cfg[branch]['pretrained']:
                raise NotImplementedError("flops calculation not implemented for beit")
            elif 'perceiver' in cfg[branch]['pretrained']:
                raise NotImplementedError("flops calculation not implemented for uni-perceiver")
            
            for key in branch_pop_keys:
                cfg[branch].pop(key, None)
            cfg[branch]["use_cls_token"] = False
            cfg[branch]["img_size"] = cfg[branch]["real_size"]
            # cfg[branch].pop("real_size")  # 为什么不把这个留下？    
            
            # # cfg[branch]["model_type"] = "augreg" 
            # if "is_branch1_deit" in cfg[branch]:
            #     cfg[branch]["model_type"] = "deit" if cfg[branch]["is_branch1_deit"] else "augreg"
            # if "deit" in cfg[branch]["pretrained"]:
            #     cfg[branch]["model_type"] = "deit"
    
    cfg.pop("output_dtype", None)
    cfg.pop("out_indices", None)
    cfg.pop("with_fpn", None)
    if cfg["type"] in [vit_models, InternViT6B]:
        if "upernet" in config_name:  # True
            pass
        elif "mask_rcnn" in config_name:
            cfg["img_size"] = 1024
        else:
            pass
        cfg["use_cls_token"] = False
        
    return cfg


def read_config(config_path):
    # if config_path.startswith("configs/"):
    #     config_path = config_path[8:]  # PIIP-1-GF2-piip3branch-vit-sbl-upernet-640_512_384-消融PIIP效果-不要数据增强.py

    # assert config_path.endswith(".py") and os.path.exists("configs/" + config_path)
    # config_file = __import__("configs." + config_path.replace(".py", "").replace("/", "."), fromlist=[''])
    
    # import pdb; pdb.set_trace()
    config_file = Config.fromfile(config_path)

    config_name = os.path.basename(config_path)
    
    cfg = config_file.model["backbone"]

    if "type" in cfg:
        cfg = convert_mmdet_config(cfg, config_name)
    
    if "piip_3branch" in config_name or cfg.get("type") == PIIPThreeBranch:
        img_size = cfg["branch3"]["img_size"]  # 图像的实际尺寸
        model_cls = PIIPThreeBranch
    # elif "piip_4branch" in config_name or cfg.get("type") == "PIIPFourBranch":
    #     img_size = cfg["branch4"]["img_size"]
    #     model_cls = PIIPFourBranch
    elif "piip_2branch" in config_name or cfg.get("type") == PIIPTwoBranch:
        img_size = cfg["branch2"]["img_size"]
        model_cls = PIIPTwoBranch
    elif "augreg" in config_name or "internvit" in config_name:
        img_size = cfg["img_size"]
        model_cls = InternViT6B
    elif "deit" in config_name or cfg.get("type") == vit_models:
        img_size = cfg["img_size"]
        model_cls = vit_models
    elif "vit" in config_name or cfg.get("type") == vit_models:
        img_size = cfg["img_size"]
        model_cls = vit_models
    else:
        raise NotImplementedError
    
    if hasattr(config_file, "input_size"):
        print(f"!!! set input size to {config_file.input_size}")
        img_size = config_file.input_size
    
    cfg.pop("type", None)
    return cfg, img_size, model_cls
