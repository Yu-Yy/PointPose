# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import argparse
import os
from tqdm import tqdm
from prettytable import PrettyTable
import copy

import _init_paths
from core.config import config
from core.config import update_config
from utils.utils import create_logger, load_backbone_panoptic
import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    return args


def main():
    args = parse_args()
    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'eval_map')
    # final_output_dir = "/home/panzhiyu/project/3d_pose/voxelpose-pytorch/output/panoptic/multi_person_posenet_50/prn64_cpn80x80x20_960x512_cam5/" #using the panotic training model
    # final_output_dir = "/home/panzhiyu/project/3d_pose/voxelpose-pytorch/output/shelf_synthetic/multi_person_posenet_50/prn64_cpn80x80x20/" # using the weight of shelf to eval the campus dataset
    # final_output_dir = "/home/panzhiyu/project/3d_pose/voxelpose-pytorch/output/campus_synthetic/multi_person_posenet_50/prn64_cpn80x80x20/"
    # final_output_dir = "/home/panzhiyu/project/3d_pose/voxelpose-pytorch/output_new/panoptic/multi_person_posenet_50/new/"
    cfg_name = os.path.basename(args.cfg).split('.')[0]

    gpus = [int(i) for i in config.GPUS.split(',')]
    print('=> Loading data ..')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    test_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
        config, config.DATASET.TEST_SUBSET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
 
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)

    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    print('=> Constructing models ..')
    model = eval('models.' + config.MODEL + '.get_multi_person_pose_net')(
        config, is_train=True)
    with torch.no_grad():
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    test_model_file = os.path.join(final_output_dir, config.TEST.MODEL_FILE)
    if config.TEST.MODEL_FILE and os.path.isfile(test_model_file):
        logger.info('=> load models state {}'.format(test_model_file))
        model.module.load_state_dict(torch.load(test_model_file))
    else:
        raise ValueError('Check the model file for testing!')

    model.eval()
    preds = []
    with torch.no_grad():
        for i, (inputs, targets_2d, weights_2d, targets_3d, meta, input_heatmap) in enumerate(tqdm(test_loader)):
            # (inputs, targets_2d, weights_2d, targets_3d, meta, input_heatmap) = test_dataset[220*10-1]
            # if len(input_heatmap) == 0:
            #     print("empty")
            #     assert len(input_heatmap) != 0,'f****k' # 逻辑错了 
            # if (i<=218):
            #     continue

            if 'panoptic' in config.DATASET.TEST_DATASET:
                # pred, _, _, _, _, _ = model(views=inputs, meta=meta)
                pred, _, _, _, _, _ = model(views=inputs,meta=meta) # 测试GT
            elif 'campus' in config.DATASET.TEST_DATASET:
                # pred, _, _, _, _, _ = model(views=inputs,meta=meta, input_heatmaps=None)
                pred, _, _, _, _, _ = model(views=None,meta=meta, input_heatmaps=input_heatmap)
            elif 'shelf' in config.DATASET.TEST_DATASET:
                pred, _, _, _, _, _ = model(views=None,meta=meta, input_heatmaps=input_heatmap)

            pred = pred.detach().cpu().numpy()

            # print(pred.shape) # [batch,10,15,5] 截取有效人数

            for b in range(pred.shape[0]): #按照batch 进行 合并,综合跑完全部结果，再做AP测试
                preds.append(pred[b])

        tb = PrettyTable() # 展示table
        if 'panoptic' in config.DATASET.TEST_DATASET:
            mpjpe_threshold = np.arange(25, 155, 25)
            aps, recs, mpjpe, _ = test_dataset.evaluate(preds)
            tb.field_names = ['Threshold/mm'] + [f'{i}' for i in mpjpe_threshold]
            tb.add_row(['AP'] + [f'{ap * 100:.2f}' for ap in aps])
            tb.add_row(['Recall'] + [f'{re * 100:.2f}' for re in recs])
            print(tb)
            print(f'MPJPE: {mpjpe:.2f}mm')
        else: # campus shelf的标准不一致
            actor_pcp, avg_pcp, bone_person_pcp, _ = test_dataset.evaluate(preds)
            tb.field_names = ['Bone Group'] + [f'Actor {i+1}' for i in range(len(actor_pcp))] + ['Average']
            for k, v in bone_person_pcp.items():
                tb.add_row([k] + [f'{i*100:.1f}' for i in v] + [f'{np.mean(v)*100:.1f}'])
            tb.add_row(['Total'] + [f'{i*100:.1f}' for i in actor_pcp] + [f'{avg_pcp*100:.1f}'])
            print(tb)


if __name__ == "__main__":
    main()
