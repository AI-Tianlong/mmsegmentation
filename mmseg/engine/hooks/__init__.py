# Copyright (c) OpenMMLab. All rights reserved.
from .visualization_hook import SegVisualizationHook

__all__ = ['SegVisualizationHook']


import torch

from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmseg.registry import HOOKS
from mmseg.registry import OPTIM_WRAPPER_CONSTRUCTORS
from torch.distributed.optim import ZeroRedundancyOptimizer

# torch_version = float(torch.__version__[:4])
# if torch_version >= 1.11:

@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class ZeroAdamW(ZeroRedundancyOptimizer):
    def __init__(self, params, optimizer_class=torch.optim.AdamW, **kwargs):
        super().__init__(params[0]['params'],
                            optimizer_class=optimizer_class,
                            parameters_as_bucket_view=True,
                            **kwargs)
        for i in range(1, len(params)):
            self.add_param_group(params[i])


# @HOOKS.register_module()
# class ZeroHook(Hook):
#     def __init__(self, interval):
#         self.interval = interval

#     def after_epoch(self, runner):
#         runner.optimizer.consolidate_state_dict(to=0)

#     def after_train_iter(self, runner):
#         if self.every_n_iters(runner, self.interval):
#             runner.optim_wrapper.consolidate_state_dict(to=0)


@HOOKS.register_module()
class ToBFloat16Hook(Hook):

    def before_run(self, runner):
        # import pdb; pdb.set_trace()
        runner.model.backbone.to(torch.bfloat16)
        runner.model.decode_head.to(torch.float32)
        try:
            runner.model.auxiliary_head.to(torch.float32)
        except:
            pass
        print("hook:", runner.model.backbone.dtype)


@HOOKS.register_module()
class ToFloat16Hook(Hook):

    def before_run(self, runner):
        runner.model.backbone.to(torch.float16)
        runner.model.decode_head.to(torch.float32)
        try:
            runner.model.auxiliary_head.to(torch.float32)
        except:
            pass
        try:
            runner.model.neck.to(torch.float32)
        except:
            pass
        print("hook:", runner.model.backbone.dtype)