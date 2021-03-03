import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import numpy as np
import utils.cameras as cameras
from utils.transforms import get_affine_transform as get_transform
from utils.transforms import affine_transform_pts_cuda as do_transform

class RoISampleLayer(nn.Module):
    def __init__(self, cfg, name_list, pool_size = 1):
        super(RoISampleLayer, self).__init__()
        self.num_camera = cfg.DATASET.CAMERA_NUM  
        self.space_size = cfg.MULTI_PERSON.SPACE_SIZE 
        self.space_center = cfg.MULTI_PERSON.SPACE_CENTER # center space 稍微修改即可
        self.cube_size = cfg.MULTI_PERSON.INITIAL_CUBE_SIZE
        # self.R = cfg.DATASET.R # new 
        # self.t = cfg.DATASET.t
        self.pool_size = pool_size # 1
        self.name_list = name_list
        self.RoIAlignLayer = torchvision.ops.MultiScaleRoIAlign(self.name_list, self.pool_size, 2)
        self.img_size = cfg.DATASET.IMAGE_SIZE
        self.VOXELSIZE = 0 # 单个voxel大小 # 这个要重新计算
        self.bbox_dict = []
        self.center = 0
        self.scale = 0 # transform ratio
        # self.scale = cfg.DATASET.scale # 
        # s 
        #image_sizes = [(512, 512)]
        
        #print('Initialization completed')


    def roi_sampleing(self, feature):
        # feature: list [(B, C, W, H) for i in range(N)]
        # return: list [(B, C, W, H) for ]
        batch_size = feature[0][self.name_list[0]].shape[0]
        num_channel = feature[0][self.name_list[0]].shape[1]
        roi_feature = []
        for B in range(batch_size):
            tmp_feature = []
            N = len(feature)
            for ind, f in enumerate(feature): #
                bbox = self.bbox_dict[ind]['bbox'] #ind 表示视角
                mask = self.bbox_dict[ind]['mask']
                bbox_tensor = torch.tensor(bbox)
                mask_tensor = torch.tensor(mask)
                bbox_roi = bbox_tensor.flatten(start_dim = 0, end_dim = 2).cuda(non_blocking=True).double()
                mask_roi = mask_tensor.flatten(start_dim = 0, end_dim = 2).cuda(non_blocking=True).double()
                df = {}
                for key in f.keys():
                    df[key] = f[key][B].unsqueeze(dim=0)
                #print(bbox_roi)
                forward_feature = self.RoIAlignLayer(df, [bbox_roi], [self.img_size[ind]]).view(-1, num_channel, self.pool_size * self.pool_size)
                
                tmp_feature.append(forward_feature * mask_roi.unsqueeze(axis=1).unsqueeze(axis=2).repeat(1, num_channel, self.pool_size * self.pool_size))
                #[(M,C,3,3) for i in range(N)] -> [(M,C,9) for i in range(N)]
            pooled_feature = torch.stack(tmp_feature, dim=3) #[(M,C,9,N)]
            
            M, C = pooled_feature.shape[0:2]
            
            #max pooling
            m = nn.MaxPool1d(N, stride=1)
            pool_f = torch.zeros(M,C,9)
            for i in range(self.pool_size * self.pool_size):
                pool_f[:,:,i] = m(pooled_feature[:,:,i,:]).squeeze() #[(M,C,9)]
            roi_feature.append(pool_f)
        
        return torch.stack(roi_feature, dim = 0).transpose(1,2) #[(B,C,M,9)]

    def generate_bbox_voxel3d(self, cam, points, img_size):
        #R: [3,3]
        #t: [3,1]
        #points: [8,3] 
        #print(R.shape)
        #print(t.shape)
        #print(points.shape)

        # AR = np.dot(R, points.T)
        # At = np.repeat(t, 8, axis = 1)
        # Ak = AR + At # [3,8]
        # points_u = Ak[0, :] / Ak[2, :] #[8,]
        # points_v = Ak[1, :] / Ak[2, :] #[8,]
        points = torch.tensor(points,dtype=torch.float)
        xy = cameras.project_pose(points, cam)
        xy = torch.clamp(xy, -1.0, max(self.img_size[0], self.img_size[1]))
        trans = torch.as_tensor( # 现实3D-> 数据 2D -> 网络2D
                        get_transform(self.center, self.scale, 0, self.img_size),
                        dtype=torch.float) # device = device
        xy = do_transform(xy, trans)
        xy = xy.numpy() # 重新对应好xy 坐标
        # trans -> new 2d coordinate
        min_u = np.min(xy[:,0]) # 
        max_u = np.max(xy[:,0])
        points_v = xy[:,1]
        points_v.sort()
        min_v = np.min(points_v)
        mid_v = sum(points_v[-4:])/4
        if min_u < 0 or max_u > img_size[0] or min_v < 0 or mid_v > img_size[1]:
            return [0,100,0,100], False
        else:
            return [min_u, min_v, max_u, mid_v], True

    def generate_bbox_Rt(self, cam, ind = 0):
        grid1Dx = np.linspace(-self.space_size[0] / 2, self.space_size[0] / 2, self.cube_size[0]) + self.space_center[0]
        grid1Dy = np.linspace(-self.space_size[1] / 2, self.space_size[1] / 2, self.cube_size[1]) + self.space_center[1]
        grid1Dz = np.linspace(-self.space_size[2] / 2, self.space_size[2] / 2, self.cube_size[2]) + self.space_center[2]
        mesh_grid = np.meshgrid(grid1Dx, grid1Dy, grid1Dz,indexing='ij')
        grid_all = mesh_grid
        
        bboxes = np.zeros([self.cube_size[0], self.cube_size[1], self.cube_size[2], 4])
        visible_mask = np.zeros([self.cube_size[0], self.cube_size[1], self.cube_size[2]])
        # R = self.R[ind]
        # t = self.t[ind]
        img_size = self.img_size # 固定image size
        for x in range(self.cube_size[0]): # using for iteration # parallel computation
            for y in range(self.cube_size[1]):
                for z in range(self.cube_size[2]):
                    points_x = grid_all[0][x,y,z]
                    points_y = grid_all[1][x,y,z]
                    points_z = grid_all[2][x,y,z]
                    points3d = np.zeros([8,3])
                    points3d[0] = [points_x - self.VOXELSIZE[0]/2, points_y - self.VOXELSIZE[1]/2, points_z - self.VOXELSIZE[2]/2]
                    points3d[1] = [points_x - self.VOXELSIZE[0]/2, points_y - self.VOXELSIZE[1]/2, points_z + self.VOXELSIZE[2]/2]
                    points3d[2] = [points_x - self.VOXELSIZE[0]/2, points_y + self.VOXELSIZE[1]/2, points_z - self.VOXELSIZE[2]/2]
                    points3d[3] = [points_x - self.VOXELSIZE[0]/2, points_y + self.VOXELSIZE[1]/2, points_z + self.VOXELSIZE[2]/2]
                    points3d[4] = [points_x + self.VOXELSIZE[0]/2, points_y - self.VOXELSIZE[1]/2, points_z - self.VOXELSIZE[2]/2]
                    points3d[5] = [points_x + self.VOXELSIZE[0]/2, points_y - self.VOXELSIZE[1]/2, points_z + self.VOXELSIZE[2]/2]
                    points3d[6] = [points_x + self.VOXELSIZE[0]/2, points_y + self.VOXELSIZE[1]/2, points_z - self.VOXELSIZE[2]/2]
                    points3d[7] = [points_x + self.VOXELSIZE[0]/2, points_y + self.VOXELSIZE[1]/2, points_z + self.VOXELSIZE[2]/2]
                    # points3d = points3d / self.scale # FROM MM TO original world coord scale
                    bboxes[x][y][z], visible_mask[x][y][z] = self.generate_bbox_voxel3d(cam, points3d, img_size)
                    
        return bboxes, visible_mask
    
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
        gridx = gridx.contiguous().view(-1, 1) # 语音与内存内容连续，与view 操作配合
        gridy = gridy.contiguous().view(-1, 1)
        gridz = gridz.contiguous().view(-1, 1)
        grid = torch.cat([gridx, gridy, gridz], dim=1) # (80*80*20), 3  这样方便直接的索引坐标
        return grid


    def forward(self, feature, meta, grid_size=None, grid_center=None, cube_size=None):

        device = feature[0][self.name_list[0]].device
        if grid_size is not None:
            self.space_size = np.array(grid_size)
        if grid_center is not None:
            self.space_center = grid_center
        if cube_size is not None:
            self.cube_size = np.array(cube_size)

        self.VOXELSIZE = self.space_size / self.cube_size # reinitialize
        
        batch_size = feature[0][self.name_list[0]].shape[0]
        num_channel = feature[0][self.name_list[0]].shape[1]
        view_num = len(feature)
        self.center = meta[0]['center'][0] # 无所谓视角与batch
        self.scale = meta[0]['scale'][0]
        if len(grid_center) == 1: # 统一的space_center
            self.space_center = grid_center[0]
            self.bbox_dict = []
            print('time start')

            for ind in range(self.num_camera): # 
                cam = {}
                for k, v in meta[ind]['camera'].items():
                    cam[k] = v[0] # 相机是固定的 batch 维度取固定就可以
                bbox, mask = self.generate_bbox_Rt(cam, ind) # 取一个视角 # 
                self.bbox_dict.append({'bbox': bbox, 'mask': mask})
            
            print('End')
            grids = self.compute_grid(self.space_size, self.space_center, self.cube_size, device=device)

            pre_volume_feature = self.roi_sampleing(feature)
            volume_feature = pre_volume_feature.squeeze(-1) #  
            volume_feature = volume_feature.view(batch_size,num_channel,cube_size[0],cube_size[1],cube_size[2])
        else: # 对space center按batch分开考虑
            nbins = cube_size[0] * cube_size[1] * cube_size[2]
            grids = torch.zeros(batch_size, nbins, 3, device=device)
            volume_feature = torch.zeros(batch_size,num_channel,nbins)
            for i in range(batch_size):
                if len(grid_center[0]) == 3 or grid_center[i][3] >= 0:
                    feature_pb = []
                    for idx in range(view_num):
                        dictf_pb = {}
                        dictf_pb[self.name_list[0]] = feature[idx][self.name_list[0]][i:i+1,:]
                        dictf_pb[self.name_list[1]] = feature[idx][self.name_list[1]][i:i+1,:]
                        feature_pb.append(dictf_pb)
                    grid = self.compute_grid(grid_size, grid_center[i], cube_size, device=device)
                    grids[i:i + 1] = grid
                    self.space_center = grid_center[i]
                    # calculate the bbox
                    self.bbox_dict = []
                    for ind in range(self.num_camera):  # N个视角重新遍历每个voxel没必要
                        cam = {}
                        for k, v in meta[ind]['camera'].items():
                            cam[k] = v[i] # 相机是固定的 batch 维度取固定就可以
                        bbox, mask = self.generate_bbox_Rt(cam, ind) # 取一个视角 # 
                        self.bbox_dict.append({'bbox': bbox, 'mask': mask})
                    # feature 
                    pre_volume_feature_pb = self.roi_sampleing(feature_pb)
                    volume_feature[i:i+1,:] = pre_volume_feature_pb.squeeze(-1)
            volume_feature = volume_feature.view(batch_size,num_channel,cube_size[0],cube_size[1],cube_size[2])

        return volume_feature, grids

    