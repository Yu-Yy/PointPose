# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.v2v_net import V2VNet
from models.project_layer import ProjectLayer
from models.roi_sample_layer import RoISampleLayer


class SoftArgmaxLayer(nn.Module):
    def __init__(self, cfg):
        super(SoftArgmaxLayer, self).__init__()
        self.beta = cfg.NETWORK.BETA

    def forward(self, x, grids):
        batch_size = x.size(0)
        channel = x.size(1)
        x = x.reshape(batch_size, channel, -1, 1)
        # x = F.softmax(x, dim=2)

        x = F.softmax(self.beta * x, dim=2)

        grids = grids.unsqueeze(1)

        x = torch.mul(x, grids)
        x = torch.sum(x, dim=2)
        return x


class PoseRegressionNet(nn.Module):
    def __init__(self, cfg):
        super(PoseRegressionNet, self).__init__()
        
        self.grid_size = cfg.PICT_STRUCT.GRID_SIZE
        self.cube_size = cfg.PICT_STRUCT.CUBE_SIZE
        self.num_joints = cfg.NETWORK.NUM_JOINTS
        self.project_layer = ProjectLayer(cfg)
        # name_list = ['feat1','feat2']
        # self.project_layer = RoISampleLayer(cfg,name_list)

        if cfg.NETWORK.FEATURE:
            self.v2v_net = V2VNet(cfg.MODEL_EXTRA.STAGE4.NUM_CHANNELS[0], cfg.NETWORK.NUM_JOINTS)
        else:
            self.v2v_net = V2VNet(cfg.NETWORK.NUM_JOINTS, cfg.NETWORK.NUM_JOINTS) # output is the number of the joints

        self.soft_argmax_layer = SoftArgmaxLayer(cfg)

    def forward(self, all_heatmaps, meta, grid_centers):
        batch_size = all_heatmaps[0].shape[0]
        # num_joints = all_heatmaps[0].shape[1]
        num_joints = self.num_joints
        device = all_heatmaps[0].device
        pred = torch.zeros(batch_size, num_joints, 3, device=device)
        cubes, grids = self.project_layer(all_heatmaps, meta,
                                          self.grid_size, grid_centers, self.cube_size) # 设置grid center位置，确定网格对应的中心点坐标


        index = grid_centers[:, 3] >= 0
        valid_cubes = self.v2v_net(cubes[index])

        pred[index] = self.soft_argmax_layer(valid_cubes, grids[index]) #

        return pred
