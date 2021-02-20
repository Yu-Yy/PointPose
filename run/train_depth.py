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
import sys

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
	    self.terminal = stream
	    self.log = open(filename, 'a')

    def write(self, message):
	    self.terminal.write(message)
	    self.log.write(message)

    def flush(self):
	    pass

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
    # 过滤为None的数据 不可有None
    # batch = list(filter(lambda x:x[0] is not None, batch))
    batch = list(filter(lambda x: None not in x, batch))
    if len(batch) == 0: return torch.Tensor()
    return default_collate(batch)



def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()
    update_config(args.cfg) # 把config的文件更新过去
 
    return args


def get_optimizer(model):
    lr = config.TRAIN.LR
    # if model.module.backbone is not None:
    #     for params in model.module.backbone.parameters():
    #         params.requires_grad = False   # If you want to train the whole model jointly, set it to be True.
    # for params in model.module.root_net.parameters():
    #     params.requires_grad = True
    # for params in model.module.pose_net.parameters():
    #     params.requires_grad = True
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=lr) # 整体模型权重均全部重新训练
    # optimizer = optim.Adam(model.module.parameters(), lr=lr)

    return model, optimizer


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
    train_dataset = eval('dataset.' + config.DATASET.TRAIN_DATASET)(config,
        config.DATASET.ROOT,config.DATASET.KP_ROOT, config.DATASET.TRAIN_VIEW_SET ,True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    # import pdb; pdb.set_trace()
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE * len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        collate_fn = ommit_collate_fn,
        pin_memory=True)

    test_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(config,
        config.DATASET.ROOT, config.DATASET.KP_ROOT, config.DATASET.TEST_VIEW_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=True,
        num_workers=config.WORKERS,
        collate_fn = ommit_collate_fn,
        pin_memory=True)

    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    print('=> Constructing models ..')
    model = eval('models.' + config.MODEL + '.build')( # hrnet_adabins.build
        config, is_train=True) # create the model
    with torch.no_grad():
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda() # 数据输送方式

    model, optimizer = get_optimizer(model) # TODO: all the parameters needs to be changed

    start_epoch = config.TRAIN.BEGIN_EPOCH
    end_epoch = config.TRAIN.END_EPOCH
    metrics_load = dict()
    best_abs_rel = 100     # TODO: load the corresponding metric using a1
    # no pretrained model loaded # not using pretrained model
    # if config.NETWORK.PRETRAINED_BACKBONE: # no pretrained test   
    #     model = load_backbone_panoptic(model, config.NETWORK.PRETRAINED_BACKBONE) # load backbone # not change
    if config.TRAIN.RESUME:
        start_epoch, model, optimizer, metrics_load = load_checkpoint_depth(model, optimizer, final_output_dir) # TODO: Load the A1 metrics
        best_abs_rel = metrics_load['abs_rel']
    

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    print('=> Training...')
    
    # generating the log
    

    # sys.stdout = Logger(stream = sys.stdout)
    for epoch in range(start_epoch, end_epoch):
        print('Epoch: {}'.format(epoch))
        # lr_scheduler.step()
        
        train_depth(config, model, optimizer, train_loader, epoch, final_output_dir, writer_dict)
        if epoch == 0:
            init_model_name =os.path.join(final_output_dir,
                                          'init_state.pth.tar')
            logger.info('saving init model state to {}'.format(
                init_model_name))
            torch.save(model.module.state_dict(), init_model_name)
        metrics = validate_depth(config, model, test_loader, final_output_dir,epoch)
        

        # get the abs_rel
        abs_rel = metrics['abs_rel']
        if abs_rel < best_abs_rel:
            best_abs_rel = abs_rel
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {} (Best: {})'.format(final_output_dir, best_model))
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.module.state_dict(),
            'metrics': metrics,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)

    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
