# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import argparse
import os
import pprint
import logging
import json
from torch.utils.data.dataloader import default_collate


import _init_paths
from depth_core.config import config
from depth_core.config import update_config
from depth_core.function import train_depth, validate_depth
from utils.utils import create_logger
from utils.utils import save_checkpoint, load_checkpoint_depth, load_model_state
from utils.utils import load_backbone_panoptic
import dataset  # a new depth dataset
import models

from dataset.panoptic_depth import Panoptic_Depth # 暂用

def ommit_collate_fn(batch):
    # import pdb; pdb.set_trace()
    # 过滤为None的数据
    batch = list(filter(lambda x:x[0] is not None, batch))
    if len(batch) == 0: return torch.Tensor()
    return default_collate(batch)



def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()
    update_config(args.cfg) # 把config的文件更新过去
 
    return args



def main():
    args = parse_args() # 读取 cfg 参数，config表示之后需要看一下
    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train') # create project log

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    gpus = [int(i) for i in config.GPUS.split(',')] # 多卡
    print('=> Loading data ..')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # RGB image processing

    test_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
        config.DATASET.ROOT,config.DATASET.VIEW_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        collate_fn = ommit_collate_fn,
        pin_memory=True)

    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    print('=> Constructing models ..')
    model = eval('models.' + config.MODEL + '.build')( # hrnet_adabins.build
        config, config.BINS, is_train=True) # create the model
    with torch.no_grad():
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda() # 数据输送方式

    test_model_file = os.path.join(final_output_dir, config.TEST.MODEL_FILE)
    if config.TEST.MODEL_FILE and os.path.isfile(test_model_file):
        logger.info('=> load models state {}'.format(test_model_file))
        model.module.load_state_dict(torch.load(test_model_file))
    else:
        raise ValueError('Check the model file for testing!')

    validate_depth(config, model, test_loader, final_output_dir, vali=True)


if __name__ == '__main__':
    main()
