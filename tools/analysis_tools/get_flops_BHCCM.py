# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import tempfile
from pathlib import Path

import torch
from mmengine import Config, DictAction
from mmengine.logging import MMLogger
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope

from mmseg.models import BaseSegmentor
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample

try:
    from mmengine.analysis import get_model_complexity_info
    # from mmengine.analysis.print_helper import _format_size
except ImportError:
    raise ImportError('Please upgrade mmengine >= 0.6.0 to use this script.')



# python tools/analysis_tools/get_flops_BHCCM.py --shape 640

# baseline
# /data/AI-Tianlong/openmmlab/mmsegmentation/configs_new/ATL-paper-new-hiera-6-20251230-修正upernet通道/baseline-GF2/part2-baseline-GF2-convnext-B-upernet-baseline-640x640.py
# /data/AI-Tianlong/openmmlab/mmsegmentation/configs_new/ATL-paper-new-hiera-6-20251230-修正upernet通道/baseline-GF2/part2-baseline-GF2-convnext-L-upernet-baseline-640x640.py
# /data/AI-Tianlong/openmmlab/mmsegmentation/configs_new/ATL-paper-new-hiera-6-20251230-修正upernet通道/baseline-GF2/part2-baseline-GF2-segnext-S-baseline-640x640.py
# /data/AI-Tianlong/openmmlab/mmsegmentation/configs_new/ATL-paper-new-hiera-6-20251230-修正upernet通道/baseline-GF2/part2-baseline-GF2-segnext-B-baseline-640x640.py
# /data/AI-Tianlong/openmmlab/mmsegmentation/configs_new/ATL-paper-new-hiera-6-20251230-修正upernet通道/baseline-GF2/part2-baseline-GF2-segnext-L-baseline-640x640.py

# /data/AI-Tianlong/openmmlab/mmsegmentation/configs_new/ATL-paper-new-hiera-6-20251230-修正upernet通道/BHCCM+LHSC-GF2/BHCCM+LHSC-GF2-convnext-B-upernet-消融6-不加relu.py
# /data/AI-Tianlong/openmmlab/mmsegmentation/configs_new/ATL-paper-new-hiera-6-20251230-修正upernet通道/BHCCM+LHSC-GF2/BHCCM+LHSC-GF2-convnext-L-upernet-消融6-不加relu.py
# /data/AI-Tianlong/openmmlab/mmsegmentation/configs_new/ATL-paper-new-hiera-6-20251230-修正upernet通道/BHCCM+LHSC-GF2/BHCCM+LHSC-GF2-segnext-S-消融6.py
# /data/AI-Tianlong/openmmlab/mmsegmentation/configs_new/ATL-paper-new-hiera-6-20251230-修正upernet通道/BHCCM+LHSC-GF2/BHCCM+LHSC-GF2-segnext-B-消融6.py
# /data/AI-Tianlong/openmmlab/mmsegmentation/configs_new/ATL-paper-new-hiera-6-20251230-修正upernet通道/BHCCM+LHSC-GF2/BHCCM+LHSC-GF2-segnext-L-消融6.py
# --shape 640


def parse_args():
    parser = argparse.ArgumentParser(
        description='Get the FLOPs of a segmentor')
    parser.add_argument('--config', help='train config file path', default=
    '/data/AI-Tianlong/openmmlab/mmsegmentation/configs_new/ATL-paper-new-hiera-6-20251230-修正upernet通道/BHCCM+LHSC-GF2/BHCCM+LHSC-GF2-segnext-S-消融6.py')
    parser.add_argument(
        '--shape', type=int, nargs='+', default=640, help='input image size')
    parser.add_argument(
    '--channel', type=int, default=4, help='input image size')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def _format_size(x: int, sig_figs: int = 3, hide_zero: bool = False) -> str:
    """Formats an integer for printing in a table or model representation.

    Expresses the number in terms of 'kilo', 'mega', etc., using
    'K', 'M', etc. as a suffix.

    Args:
        x (int): The integer to format.
        sig_figs (int): The number of significant figures to keep.
            Defaults to 3.
        hide_zero (bool): If True, x=0 is replaced with an empty string
            instead of '0'. Defaults to False.

    Returns:
        str: The formatted string.
    """
    if hide_zero and x == 0:
        return ''

    def fmt(x: float) -> str:
        # use fixed point to avoid scientific notation
        return f'{{:.{sig_figs}f}}'.format(x).rstrip('0').rstrip('.')

    # if abs(x) > 1e14:
    #     return fmt(x / 1e15) + 'P'
    # if abs(x) > 1e11:
    #     return fmt(x / 1e12) + 'T'
    if abs(x) > 1e8:
        return fmt(x / 1e9) + 'G'
    if abs(x) > 1e5:
        return fmt(x / 1e6) + 'M'
    if abs(x) > 1e2:
        return fmt(x / 1e3) + 'K'
    return str(x)

def fmt(x: float) -> str:
    # use fixed point to avoid scientific notation
    return f'{{:.{4}f}}'.format(x).rstrip('0').rstrip('.')

def inference(args: argparse.Namespace, logger: MMLogger) -> dict:
    config_name = Path(args.config)

    input_channel = args.channel

    if not config_name.exists():
        logger.error(f'Config file {config_name} does not exist')

    cfg: Config = Config.fromfile(config_name)
    cfg.work_dir = tempfile.TemporaryDirectory().name
    cfg.log_level = 'WARN'
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    init_default_scope(cfg.get('scope', 'mmseg'))

    if len(args.shape) == 1:
        input_shape = (input_channel, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (input_channel, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')
    result = {}

    model: BaseSegmentor = MODELS.build(cfg.model)
    if hasattr(model, 'auxiliary_head'):
        model.auxiliary_head = None
    if torch.cuda.is_available():
        model.cuda()
    model = revert_sync_batchnorm(model)
    result['ori_shape'] = input_shape[-2:]
    result['pad_shape'] = input_shape[-2:]
    data_batch = {
        'inputs': [torch.rand(input_shape)],
        'data_samples': [SegDataSample(metainfo=result)]
    }
    data = model.data_preprocessor(data_batch)
    model.eval()
    if cfg.model.decode_head.type in ['MaskFormerHead', 'Mask2FormerHead']:
        # TODO: Support MaskFormer and Mask2Former
        raise NotImplementedError('MaskFormer and Mask2Former are not '
                                  'supported yet.')

    outputs = get_model_complexity_info(
        model,
        # input_shape, #注释掉这里
        inputs=data['inputs'], # [1,4,640,640]
        show_table=True, # 默认False
        show_arch=True)  # 默认False
    # result['flops'] = _format_size(outputs['flops'])
    # result['params'] = _format_size(outputs['params'])
    result['flops'] = fmt(outputs['flops']/ 1e9) + 'G'
    result['params'] = fmt(outputs['params']/ 1e6) + 'M'
    result['compute_type'] = 'direct: randomly generate a picture'
    print(data['inputs'].shape)
    return result


def main():

    args = parse_args()
    logger = MMLogger.get_instance(name='MMLogger')

    result = inference(args, logger)
    split_line = '=' * 30
    ori_shape = result['ori_shape']
    pad_shape = result['pad_shape']
    flops = result['flops']
    params = result['params']

    # import pdb; pdb.set_trace()
    # flops_G = f"{result['flops'] / 1e9:.3f}G"
    # params_M = f"{result['params'] / 1e6:.3f}M"
    compute_type = result['compute_type']

    if pad_shape != ori_shape:
        print(f'{split_line}\nUse size divisor set input shape '
              f'from {ori_shape} to {pad_shape}')
    
    print(f'{split_line}')
    print(f'Compute type:{compute_type}')
    print(f'Input shape: {pad_shape}')
    print(f'Flops: {flops}')
    print(f'Params: {params}')
    print(f'{split_line}')

    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify '
          'that the flops computation is correct.')


    
    # print(f'{split_line}\nCompute type: {compute_type}\n'
    #       f'Input shape: {pad_shape}\nFlops: {flops}\n'
    #       f'Params: {params}\n{split_line}')
    # print('!!!Please be cautious if you use the results in papers. '
    #       'You may need to check if all ops are supported and verify '
    #       'that the flops computation is correct.')


if __name__ == '__main__':
    main()
