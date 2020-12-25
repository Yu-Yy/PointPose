# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils.cameras as cameras
from utils.transforms import get_affine_transform as get_transform
from utils.transforms import affine_transform_pts_cuda as do_transform


class ProjectLayer(nn.Module):
    def __init__(self, cfg):
        super(ProjectLayer, self).__init__()
        self.feature_op = cfg.NETWORK.FEATURE
        self.img_size = cfg.NETWORK.IMAGE_SIZE  # 输入图片的大小
        # self.heatmap_size = cfg.NETWORK.HEATMAP_SIZE # 
        
        self.heatmap_size = np.array([cfg.NETWORK.IMAGE_SIZE[0]/2,cfg.NETWORK.IMAGE_SIZE[1]/2],dtype=np.int16) #[cfg.NETWORK.IMAGE_SIZE[0] / 2, cfg.NETWORK.IMAGE_SIZE[1] / 2] 
        # only these different

        self.grid_size = cfg.MULTI_PERSON.SPACE_SIZE
        self.cube_size = cfg.MULTI_PERSON.INITIAL_CUBE_SIZE
        self.grid_center = cfg.MULTI_PERSON.SPACE_CENTER 

    def compute_grid(self, boxSize, boxCenter, nBins, device=None): 
        if isinstance(boxSize, int) or isinstance(boxSize, float):
            boxSize = [boxSize, boxSize, boxSize]
        if isinstance(nBins, int):
            nBins = [nBins, nBins, nBins]

        grid1Dx = torch.linspace(-boxSize[0] / 2, boxSize[0] / 2, nBins[0], device=device)
        grid1Dy = torch.linspace(-boxSize[1] / 2, boxSize[1] / 2, nBins[1], device=device)
        grid1Dz = torch.linspace(-boxSize[2] / 2, boxSize[2] / 2, nBins[2], device=device) # torch 建立网格x y z 坐标
        gridx, gridy, gridz = torch.meshgrid(
            grid1Dx + boxCenter[0], #
            grid1Dy + boxCenter[1],
            grid1Dz + boxCenter[2],
        ) # grid 建立的整体网格对应的现实空间的坐标属性
        gridx = gridx.contiguous().view(-1, 1) #语音与内存内容连续，与view 操作配合
        gridy = gridy.contiguous().view(-1, 1) 
        gridz = gridz.contiguous().view(-1, 1)
        grid = torch.cat([gridx, gridy, gridz], dim=1) # (80*80*20), 3  这样方便直接的索引坐标
        return grid

    def get_voxel(self, heatmaps, meta, grid_size, grid_center, cube_size):
        device = heatmaps[0].device
        batch_size = heatmaps[0].shape[0] # 按照batch size 
        channel_num = heatmaps[0].shape[1] # 取得channel维度数目
        nbins = cube_size[0] * cube_size[1] * cube_size[2] # 80 * 80 * 20
        n = len(heatmaps) # n 为视角数目
        cubes = torch.zeros(batch_size, channel_num, 1, nbins, n, device=device)
        # h, w = heatmaps[0].shape[2], heatmaps[0].shape[3]
        w, h = self.heatmap_size # this size is ?? for the heatmap size
        grids = torch.zeros(batch_size, nbins, 3, device=device) # grid建立
        bounding = torch.zeros(batch_size, 1, 1, nbins, n, device=device)
        for i in range(batch_size):
            if len(grid_center[0]) == 3 or grid_center[i][3] >= 0:
                # This part of the code can be optimized because the projection operation is time-consuming.
                # If the camera locations always keep the same, the grids and sample_grids are repeated across frames   # 固定相机可以只计算一次
                # and can be computed only one time.
                if len(grid_center) == 1: 
                    grid = self.compute_grid(grid_size, grid_center[0], cube_size, device=device) # 只考虑这里 grid center 可以 传一个list || 返回mm 单位点的实际空间对应的voxel坐标
                else:
                    grid = self.compute_grid(grid_size, grid_center[i], cube_size, device=device)
                grids[i:i + 1] = grid # let grids in the batch size # 同一空间下，这个只计算一次就可以 
                for c in range(n):
                    center = meta[c]['center'][i] #center & scale is consistent
                    scale = meta[c]['scale'][i] 

                    width, height = center * 2 # 恢复原图的w与h
                    trans = torch.as_tensor( # 现实3D-> 数据 2D -> 网络2D
                        get_transform(center, scale, 0, self.img_size),
                        dtype=torch.float,
                        device=device)
                    cam = {}
                    for k, v in meta[c]['camera'].items():
                        cam[k] = v[i]
                    xy = cameras.project_pose(grid, cam) # project all the 3D points into 2D 

                    bounding[i, 0, 0, :, c] = (xy[:, 0].float() >= 0) & (xy[:, 1].float() >= 0) & (xy[:, 0].float() < width.float()) & (
                                xy[:, 1].float() < height.float()) # 
                    xy = torch.clamp(xy, -1.0, max(width, height)) # N*2 coordinates to clamp into the setting range
                    xy = do_transform(xy, trans) # trans for the xy coord
                    xy = xy * torch.tensor(
                        [w, h], dtype=torch.float, device=device) / torch.tensor(
                        self.img_size, dtype=torch.float, device=device) # xy 坐标对应到heatmap 坐标上
                    sample_grid = xy / torch.tensor(
                        [w - 1, h - 1], dtype=torch.float,
                        device=device) * 2.0 - 1.0   # 归一化 ，中心放在0 [-1,1]
                    sample_grid = torch.clamp(sample_grid.view(1, 1, nbins, 2), -1.1, 1.1) 

                    # if pytorch version < 1.3.0, align_corners=True should be omitted.
                    cubes[i:i + 1, :, :, :, c] += F.grid_sample(heatmaps[c][i:i + 1, :, :, :], sample_grid, align_corners=True) #  以这个sample进行采样 将坐标规范到-1，1的区间内

        # cubes = cubes.mean(dim=-1)
        cubes = torch.sum(torch.mul(cubes, bounding), dim=-1) / (torch.sum(bounding, dim=-1) + 1e-6) # mul为点乘
        cubes[cubes != cubes] = 0.0
        if not self.feature_op:
            cubes = cubes.clamp(0.0, 1.0)  # cubes 的处理 可之后再考虑 # for heatmap process recover this

        cubes = cubes.view(batch_size, channel_num, cube_size[0], cube_size[1], cube_size[2])  ## 采样每个视角的2D 结果
        return cubes, grids

    def forward(self, heatmaps, meta, grid_size, grid_center, cube_size):
        cubes, grids = self.get_voxel(heatmaps, meta, grid_size, grid_center, cube_size)
        return cubes, grids # 采样后的3D voxel 以及其对应的实际物理世界的坐标