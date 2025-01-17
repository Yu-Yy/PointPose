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

# temporal 
# from models.voteposenet import VotePoseNet # temporal setting

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
from depth_core.function import train_points3d, validate_points3d
from utils.utils import create_logger
from utils.utils import save_checkpoint, load_checkpoint_point3d, load_model_state
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
    for params in model.module.hr_adbins.parameters():
        params.requires_grad = False  # do not grad the 2d parts
    for params in model.module.votepose.parameters():
        params.requires_grad = True
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=lr) # 整体模型权重均全部重新训练
    # optimizer = optim.Adam(model.module.parameters(), lr=lr)

    return model, optimizer


def main():
    
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

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
        shuffle=True, # still true
        num_workers=config.WORKERS,
        collate_fn = ommit_collate_fn,
        pin_memory=True)

    print('=> Constructing models ..')
    model = eval('models.' + config.MODEL)( # hrnet_adabins.build
        config, is_train=True) # create the model

    with torch.no_grad():
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda() # 数据输送方式

    model, optimizer = get_optimizer(model) # TODO: all the parameters needs to be changed
    

    start_epoch = config.TRAIN.BEGIN_EPOCH
    end_epoch = config.TRAIN.END_EPOCH
    # model_file = '/home/panzhiyu/project/3d_pose/voxelpose-pytorch/output_multitask_2d_depth_5cam/panoptic_depth/hrnet_adabins_50/depth_est/model_best.pth.tar'
    
    if config.PRETRAINED:
        model_file = '/home/panzhiyu/project/3d_pose/voxelpose-pytorch/point_3dpose/panoptic_depth_multview/voteposenet_50/pose3d_est/temp_model.pth.tar'
        logger.info('=> load models pretrained {}'.format(model_file))
        pretrained_model = torch.load(model_file)
        model.module.load_state_dict(pretrained_model)

    # mini_loss_3d = 100 # a high value
    # load from previous training
    if config.TRAIN.RESUME:
        start_epoch, model, optimizer, metrics_load = load_checkpoint_point3d(model, optimizer, final_output_dir) # TODO: Load the A1 metrics
        mini_loss_3d = metrics_load


    # if config.TRAIN.RESUME: 
    #     logger.info('=> load models state {}'.format(model_file))
    #     pretrained_state_dict = torch.load(model_file) # changed the unchanged part delete that weight
    #     ## only load the match part
    #     # pretrained_state_dict = torch.load(pretrained_file)
    #     model_state_dict = model.module.hr_adbins.state_dict()
    #     prefix = "module."
    #     # print('orig')
    #     # print(model_state_dict.keys())
    #     # print('pretrained')
    #     # print(pretrained_state_dict.keys())
    #     # error
    #     new_pretrained_state_dict = {}
    #     for k, v in pretrained_state_dict.items():
    #         if k.replace(prefix, "") in model_state_dict and v.shape == model_state_dict[k.replace(prefix, "")].shape:
    #             new_pretrained_state_dict[k.replace(prefix, "")] = v
    #     model.module.hr_adbins.load_state_dict(new_pretrained_state_dict) # no redundent parameters

    # Load the pretrained model

    

    # load pretrained model
    # metrics_load = dict()
    # best_abs_rel = 100     # TODO: load the corresponding metric using a1
    # no pretrained model loaded # not using pretrained model
    # if config.NETWORK.PRETRAINED_BACKBONE: # no pretrained test   
    #     model = load_backbone_panoptic(model, config.NETWORK.PRETRAINED_BACKBONE) # load backbone # not change
    # model_file = '/home/panzhiyu/project/3d_pose/voxelpose-pytorch/output_multitask_2d_depth_5cam/panoptic_depth/hrnet_adabins_50/depth_est/model_best.pth.tar'
    # if config.TRAIN.RESUME:
    #     logger.info('=> load models state {}'.format(model_file))
    #     # model.module.init_weight(model_file)
    #     # model.module.load_state_dict(torch.load(model_file))
    #     # start_epoch, model, optimizer, metrics_load = load_checkpoint_depth(model, optimizer, final_output_dir) # TODO: Load the A1 metrics
    #     # best_abs_rel = metrics_load['abs_rel']
    

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
        train_points3d(config, model, optimizer, train_loader, epoch, final_output_dir, writer_dict, logger=logger)
        if epoch % 2 ==  0:
            model_name =os.path.join(final_output_dir,
                                          f'epoch{epoch}_state.pth.tar')
            logger.info('saving current model state to {}'.format(
                model_name))
            torch.save(model.module.state_dict(), model_name)
        loss_3d_c = validate_points3d(config, model, test_loader, final_output_dir,epoch,logger=logger)
        
        # # get the abs_rel
        if loss_3d_c < mini_loss_3d:
            mini_loss_3d = loss_3d_c
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {} (Best: {})'.format(final_output_dir, best_model))
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.module.state_dict(),
            'loss_3d': mini_loss_3d,
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
