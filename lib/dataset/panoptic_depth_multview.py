# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import glob
import os.path as osp
import numpy as np
# import json_tricks as json 
import json
import pickle
import logging
import os 
import copy
from torch.utils.data import Dataset
import cv2
import torch
import torchvision.transforms as transforms
from functools import reduce
import matplotlib.pyplot as plt 

# GEN_LIST = [
# "160226_haggling1",
# "160906_ian1",
# "160906_ian2",
# "160906_band1",
# "160906_band2",
# "160906_pizza1",
# "160422_haggling1",
# "160906_ian5",
# ]

# train the dataset in the 2D's test set
GEN_LIST = ["161202_haggling1",
            "160906_ian3", 
]

TEST_LIST = [
    # "161202_haggling1",
    # "160906_ian3",
    "160906_band3",
]



class Panoptic_Depth_Mul(Dataset):
    def __init__(self, cfg, image_folder, keypoint_folder, view_set,is_train = True, transform = None): # TODO add keypoint folder  cfg
        self.view_set = view_set
        self.view_num = len(self.view_set)
        self.image_folder = image_folder
        self.transform = transform
        # self.overall_view = [1,2,3,4,5]
        self.depth_size = np.array([512,424])
        self.image_size = np.array([1920,1080])
        # self.input_size = cfg.dataset.input_size 
        self.input_size = np.array([960,512])
        self.crop_size = np.array([680,512])
        self.heatmap_size = self.input_size / 2
        self.heatmap_size = self.heatmap_size.astype(np.int16)
        self.num_joints = cfg.NETWORK.NUM_JOINTS
        self.sigma = cfg.NETWORK.SIGMA
        self.max_people = cfg.DATASET.MAX_PEOPLE_NUM
        self.single_size = 512*424
        if is_train:
            self.scene_list = GEN_LIST
        else:
            self.scene_list = TEST_LIST
        # 读取k_calibration, ksync, 以depth 为准对齐一次即可
        self.scene_num = len(self.scene_list)
        # 以List 形式读取文件
        self.kcalib_data = []
        self.ksync_data = []
        self.sync_data = []
        self.calib_data = []
        for scene in self.scene_list:
            with open(os.path.join(image_folder,scene,f'kcalibration_{scene}.json'),'rb') as dfile:
                self.kcalib_data.append(json.load(dfile))
        for scene in self.scene_list:
            with open(os.path.join(image_folder,scene,f'ksynctables_{scene}.json'),'rb') as dfile:
                self.ksync_data.append(json.load(dfile))
        for scene in self.scene_list:
            with open(os.path.join(image_folder,scene,f'calibration_{scene}.json'),'rb') as dfile:
                self.calib_data.append(json.load(dfile))
        for scene in self.scene_list:
            with open(os.path.join(image_folder,scene,f'synctables_{scene}.json'),'rb') as dfile:
                self.sync_data.append(json.load(dfile))
        # calculate the total frame idx for the specific idx
        # in kp form, all the other elements needs to be not None
        self.kp3d_list = [osp.join(keypoint_folder, x, 'hdPose3d_stage1_coco19') for x in self.scene_list]
        self.anno_kp_files = []
        self.num_pers = []
        for kp_anno in self.kp3d_list:
            self.anno_kp_files.append(sorted(glob.iglob('{:s}/*.json'.format(kp_anno)))) 
            self.num_pers.append(len(self.anno_kp_files[-1]))

        self.num_pers = np.array(self.num_pers)
        self.until_sum = np.cumsum(self.num_pers)

        # create the scaling matrix

        self.scale_kinoptic2panoptic = np.eye(4)
        scaleFactor = 100
        self.scale_kinoptic2panoptic[0:3,0:3] = scaleFactor * self. scale_kinoptic2panoptic[0:3,0:3]
        
        # for scene in self.scene_list:
        #     for view in self.view_set:
        #         fsize = os.path.getsize(os.path.join(self.image_folder,scene,'kinect_shared_depth', f'KINECTNODE{view}','depthdata.dat'))
        #         self.num_pers.append(fsize/(2*self.single_size))
        
        
    def __len__(self):
        return int(np.sum(self.num_pers))

    def __getitem__(self,index): # return image, depth, valid_mask, hm, weights, trans_matrix, K, M, diff  numpy form by view
        # confirm the indexing scene and corresponding view
        # import pdb
        # pdb.set_trace()
        findtable = self.until_sum - (index + 1)
        scene_index = np.min(np.where(findtable>=0)[0])
        until_num = np.sum(self.num_pers[:scene_index])
        kp_idx = index - until_num
        # read the kp data and process in different view
        with open(self.anno_kp_files[scene_index][kp_idx],'rb') as kp: 
            try:
                kp_row_data = json.load(kp)
            except:
                return None,None,None,None,None,None,None,None,None, None

        univ_time = kp_row_data['univTime']  
        kp3d_body_data = kp_row_data['bodies'] 
        num_people = len(kp3d_body_data)
        if num_people == 0:
            return None,None,None,None,None,None,None,None,None, None
        # get the 3d
        # output_pose_3d = torch.zeros(self.max_people, self.num_joints, 4)
        pose_3d = []
        for n in range(num_people):
            pose3d = np.array(kp3d_body_data[n]['joints19']).reshape((-1, 4))
            pose_3d.append(np.expand_dims(pose3d,axis=0))
        pose_3d = np.concatenate(pose_3d,axis=0) # (N, 19, 4)
        pose_3d = torch.from_numpy(pose_3d)
        offset = self.max_people - num_people
        fill_indx = torch.randint(num_people, (offset, ))
        fill_tensor = pose_3d[fill_indx,:]
        output_pose_3d = torch.cat([pose_3d,fill_tensor],dim=0) # generate the gt pose replicated

        # output_pose_3d[:num_people,...] = torch.from_numpy(pose_3d)
            
        # process in views
        output_img = []
        output_depth = []
        output_valid_mask = []
        output_hm = []
        output_weights = []
        output_trans = []
        output_K = []
        output_M = []
        output_diff = []
        for view in self.view_set: # not the idx form
            # match rgb
            match_synctable_rgb = np.array(self.ksync_data[scene_index]['kinect']['color'][f'KINECTNODE{int(view)}']['univ_time']) - 6.25
            target_frame_rgb_idx = np.argmin(abs(match_synctable_rgb - univ_time),axis=0)
            if abs(self.ksync_data[scene_index]['kinect']['color'][f'KINECTNODE{int(view)}']['univ_time'][target_frame_rgb_idx] - 6.25 - univ_time) > 30:
                return None, None, None, None, None, None, None, None, None, None # not yet
            match_synctable_depth = np.array(self.ksync_data[scene_index]['kinect']['depth'][f'KINECTNODE{int(view)}']['univ_time'])
            target_frame_depth_idx = np.argmin(abs(match_synctable_depth - univ_time),axis=0)
            if abs(self.ksync_data[scene_index]['kinect']['depth'][f'KINECTNODE{int(view)}']['univ_time'][target_frame_depth_idx] - univ_time) > 17:
                return None, None, None, None, None, None, None, None, None, None # not yet
            if abs(self.ksync_data[scene_index]['kinect']['color'][f'KINECTNODE{int(view)}']['univ_time'][target_frame_rgb_idx] - self.ksync_data[scene_index]['kinect']['depth'][f'KINECTNODE{int(view)}']['univ_time'][target_frame_depth_idx]) > 6.5:
                return None, None, None, None, None, None, None, None, None, None # not yet

            # get the trans matrix
            panoptic_calibData = self.calib_data[scene_index]['cameras'][view + 509] # from 510 to 519
            M = np.concatenate([np.array(panoptic_calibData['R']),np.array(panoptic_calibData['t'])],axis=-1)
            T_panopticWorld2KinectColor = np.concatenate([M,np.array([[0,0,0,1]])],axis=0)
            T_kinectColor2PanopticWorld = np.linalg.pinv(T_panopticWorld2KinectColor)

            kcalibdata = self.kcalib_data[scene_index]['sensors'][view - 1]
            T_kinectColor2KinectLocal = np.array(kcalibdata['M_color'])
            T_kinectLocal2KinectColor = np.linalg.pinv(T_kinectColor2KinectLocal)
            T_kinectLocal2PanopticWorld = T_kinectColor2PanopticWorld @ self.scale_kinoptic2panoptic @ T_kinectLocal2KinectColor
            T_kinectLocal2PanopticWorld = np.expand_dims(T_kinectLocal2PanopticWorld, axis = 0)
            output_trans.append(torch.from_numpy(T_kinectLocal2PanopticWorld))

            K_color = np.array(kcalibdata['K_color'])
            K_color = np.expand_dims(K_color,axis=0)
            output_K.append(torch.from_numpy(K_color))
            K_diff = np.zeros(12)
            temp = np.array(kcalibdata['distCoeffs_color'])
            temp = temp.reshape(-1)
            K_diff[:5] = temp[:5].copy()
            K_diff = np.expand_dims(K_diff,axis=0)
            output_diff.append(torch.from_numpy(K_diff))
            M_color = np.array(kcalibdata['M_color'])
            M_color = np.expand_dims(M_color,axis=0)
            output_M.append(torch.from_numpy(M_color))

            # process the depth data
            fdepth = open(os.path.join(self.image_folder,self.scene_list[scene_index],'kinect_shared_depth',f'KINECTNODE{int(view)}','depthdata.dat'),'rb')
            fdepth.seek(2*self.single_size*(target_frame_depth_idx), os.SEEK_SET)
            depth_org = np.fromfile(fdepth, count = self.single_size, dtype=np.int16)
            depth_org = depth_org.reshape([self.depth_size[1],self.depth_size[0]])
            depth_org = depth_org[...,::-1] * 0.001 # change in meter 
            depth_proc = depth_org.reshape(-1,1)
            
            # unproject the data and project in color plane
            point_3D, point_3d = self.__unprojectPoints__(kcalibdata, depth_org)

            # project the 3d point into 2D
            K_color = np.array(kcalibdata['K_color'])
            M_color = np.array(kcalibdata['M_color'])
            R = M_color[:3,:3]
            t = M_color[:3,3:4]
            Kd = np.array(kcalibdata['distCoeffs_color'])
            color_2d, depth_val = self.__projectPoints__(point_3D[:3,:],K_color,R,t,Kd)
            color_2d = color_2d[:2,:].transpose() # the color plane coordinate
            # mapping the coordinate into the map
            depth_c = np.zeros([self.image_size[1],self.image_size[0]])
            color_map = color_2d.astype(np.int16)
            color_map[:,0] = np.clip(color_map[:,0], 0, self.image_size[0]-1)
            color_map[:,1] = np.clip(color_map[:,1], 0, self.image_size[1]-1)

            depth_c[color_map[:,1:2],color_map[:,0:1]] = depth_proc # not the filling

            img_frame = cv2.imread(osp.join(self.image_folder,self.scene_list[scene_index],'kinectImgs',f'50_{int(view):0>2d}',f'50_{int(view):0>2d}_{int(target_frame_rgb_idx + 1):0>8d}.jpg'),
                                cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

            # do the transform
            height, width, _ = img_frame.shape
            c = np.array([width / 2.0, height / 2.0]) # 
            s = self.__get_scale__((width, height), self.input_size) # (960 * 512) # 读入和实际
            r = 0

            trans = self.__get_affine_transform__(c, s, r, self.input_size)  # 2D GT 做同样的变换 # 表示平移和旋转关系

            img = cv2.warpAffine(
                    img_frame,
                    trans, (int(self.input_size[0]), int(self.input_size[1])),
                    flags=cv2.INTER_LINEAR)
            # cv2.imwrite(f'debug_rgb_{view}.jpg',img)
            depth_out = cv2.warpAffine(
                            depth_c,
                            trans, (int(self.input_size[0]), int(self.input_size[1])),
                            flags=cv2.INTER_LINEAR)
            img = img[:,141:821]
            # do transform
            if self.transform:
                img = self.transform(img)
            img = img.unsqueeze(0)
            output_img.append(img)
            depth_out = depth_out[:,141:821]

            depth_out = torch.from_numpy(depth_out)
            depth_out = depth_out.unsqueeze(0)
            depth_out = depth_out.unsqueeze(0)
            output_depth.append(depth_out)
            # mask_out = depth_out > 0.01
            # read in the mask
            try:
                mask_gt = cv2.imread(osp.join(self.image_folder,self.scene_list[scene_index],'hd_mask',f'50_{int(view):0>2d}',f'50_{int(view):0>2d}_{int(target_frame_rgb_idx + 1):0>8d}.jpg'), cv2.IMREAD_GRAYSCALE)
                mask_gt = cv2.warpAffine(
                                mask_gt,
                                trans, (int(self.input_size[0]), int(self.input_size[1])),
                                flags=cv2.INTER_LINEAR)
                # mask_gt = torch.from_numpy(mask_gt)
                mask_gt = mask_gt[:,141:821]
                t_mask = transforms.ToTensor()
                mask_gt = t_mask(mask_gt)
                # import pdb; pdb.set_trace()
                output_valid_mask.append(mask_gt.unsqueeze(0))
            except:
                # import pdb; pdb.set_trace()
                #output_valid_mask.append(torch.ones(1,int(self.crop_size[1]),int(self.crop_size[0]))) # all one for mask
                return None, None, None, None, None, None, None, None, None, None # 舍弃问题mask

            # process the heatmap
            joints = []
            joints_vis = []
            for n in range(num_people): # TODO: using batch process for accelerating
                pose3d = np.array(kp3d_body_data[n]['joints19']).reshape((-1, 4))
                anno_vis = pose3d[:, -1] > 0.1
                pose3d_proc = pose3d[...,:3]
                R_p = np.array(panoptic_calibData['R'])
                T_p = np.array(panoptic_calibData['t'])
                pose2d = self.__projectjointsPoints__(pose3d_proc.transpose(),K_color,R_p, T_p, Kd)
                pose2d = pose2d[:2,:].transpose()
                x_check = np.bitwise_and(pose2d[:, 0] >= 0, 
                                            pose2d[:, 0] <= self.image_size[0] - 1) #(15,) bool
                y_check = np.bitwise_and(pose2d[:, 1] >= 0,
                                            pose2d[:, 1] <= self.image_size[1] - 1)
                check = np.bitwise_and(x_check, y_check) # check bool se
                anno_vis[np.logical_not(check)] = 0
                # process the joints
                for i in range(len(pose2d)):
                    pose2d[i, 0:2] = self.__affine_transform__(  # joints 为 GT 处理的结果
                        pose2d[i, 0:2], trans)
                    if (np.min(pose2d[i, :2]) < 0 or
                            pose2d[i, 0] >= self.input_size[0] or
                            pose2d[i, 1] >= self.input_size[1]):
                        anno_vis[i] = 0
                joints.append(pose2d)
                anno_vis = np.expand_dims(anno_vis,axis = -1)
                joints_vis.append(anno_vis)

            target_heatmap, target_weight = self.__generate_target_heatmap__(
                joints, joints_vis)
            target_heatmap = target_heatmap[:,:,71:411]
            target_heatmap = torch.from_numpy(target_heatmap)
            output_hm.append(target_heatmap.unsqueeze(0))
            target_weight = torch.from_numpy(target_weight)
            output_weights.append(target_weight.unsqueeze(0))

        output_img = torch.cat(output_img,dim=0)
        output_depth = torch.cat(output_depth,dim=0)
        output_valid_mask = torch.cat(output_valid_mask,dim=0)
        output_hm = torch.cat(output_hm,dim=0)
        output_weights = torch.cat(output_weights,dim=0)
        output_trans = torch.cat(output_trans,dim=0)
        output_K = torch.cat(output_K, dim=0)
        output_M = torch.cat(output_M, dim=0)
        output_diff = torch.cat(output_diff,dim=0)
            
        return output_img, output_depth, output_valid_mask,output_hm, output_weights, output_trans, output_K, output_M, output_diff, output_pose_3d

    def __affine_transform__(self, pt, t):
        new_pt = np.array([pt[0], pt[1], 1.]).T
        new_pt = np.dot(t, new_pt)
        return new_pt[:2]

    def __unprojectPoints__(self,kcalibdata,depth_org):
        x_cor, y_cor = np.meshgrid(range(self.depth_size[0]), range(self.depth_size[1]))
        x_cor = x_cor.reshape(-1,1)
        y_cor = y_cor.reshape(-1,1)
        cor_2d = np.concatenate([x_cor,y_cor,np.ones([x_cor.shape[0],1])],axis=-1).transpose()
        K_depth = np.array(kcalibdata['K_depth'])
        norm_2d = np.linalg.pinv(K_depth) @ cor_2d
        norm_2d = norm_2d.transpose()
        x_cor_depth = norm_2d[:,0:1]
        x_cor_bak = x_cor_depth.copy()
        y_cor_depth = norm_2d[:,1:2]
        y_cor_bak = y_cor_depth.copy()
        K_diff = np.zeros(12)
        temp = np.array(kcalibdata['distCoeffs_depth'])
        K_diff[:5] = temp[:5].copy()
        # undistortion
        for _ in range(5):
            r2 = x_cor_depth * x_cor_depth + y_cor_depth * y_cor_depth
            icdist = (1 + ((K_diff[7]*r2 + K_diff[6])*r2 + K_diff[5])*r2) / (1 + ((K_diff[4]*r2 + K_diff[1])*r2 + K_diff[0])*r2)
            deltaX = 2*K_diff[2] *x_cor_depth *y_cor_depth + K_diff[3]*(r2 + 2*x_cor_depth * x_cor_depth)+  K_diff[8]*r2+K_diff[9]* r2 *r2
            deltaY = K_diff[2]*(r2 + 2*y_cor_depth *y_cor_depth) + 2*K_diff[3]*x_cor_depth *y_cor_depth+ K_diff[10] * r2 + K_diff[11]* r2 *r2

            x_cor_depth = (x_cor_bak - deltaX) *icdist
            y_cor_depth = (y_cor_bak - deltaY) *icdist

        depth_proc = depth_org.reshape(-1,1) 
        x_cor_depth = x_cor_depth * depth_proc
        y_cor_depth = y_cor_depth * depth_proc
        depth_cam = np.concatenate([x_cor_depth,y_cor_depth,depth_proc,np.ones(x_cor_depth.shape)],axis=-1)
        M_depth = np.array(kcalibdata['M_depth'])
        point_3D = np.linalg.pinv(M_depth) @ depth_cam.transpose()
        point_3d = point_3D[:3,:].transpose()

        return point_3D, point_3d
    
    def __projectPoints__(self, X, K, R, t, Kd):
        """
        Projects points X (3xN) using camera intrinsics K (3x3),
        extrinsics (R,t) and distortion parameters Kd=[k1,k2,p1,p2,k3].
        Roughly, x = K*(R*X + t) + distortion
        See http://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
        or cv2.projectPoints
        """

        x = np.dot(R, X) + t  # panoptic to kinect color scaling
        depth_val = x[2:3,:].transpose()

        x[0:2, :] = x[0:2, :] / (x[2, :] + 1e-5)

        r = x[0, :] * x[0, :] + x[1, :] * x[1, :]

        # 去畸变
        x[0, :] = x[0, :] * (1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r
                            ) + 2 * Kd[2] * x[0, :] * x[1, :] + Kd[3] * (
                                r + 2 * x[0, :] * x[0, :])
        x[1, :] = x[1, :] * (1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r
                            ) + 2 * Kd[3] * x[0, :] * x[1, :] + Kd[2] * (
                                r + 2 * x[1, :] * x[1, :])

        x[0, :] = K[0, 0] * x[0, :] + K[0, 1] * x[1, :] + K[0, 2]
        x[1, :] = K[1, 0] * x[0, :] + K[1, 1] * x[1, :] + K[1, 2]

        return x, depth_val

    def __get_scale__(self, image_size, resized_size): # confirm the equal_scale transform
        w, h = image_size
        w_resized, h_resized = resized_size  # no padding
        w_pad = w
        h_pad = h
        # if w / w_resized < h / h_resized:
        #     w_pad = h / h_resized * w_resized
        #     h_pad = h
        # else:
        #     w_pad = w
        #     h_pad = w / w_resized * h_resized
        scale = np.array([w_pad / 200.0, h_pad / 200.0], dtype=np.float32)

        return scale

    def __projectjointsPoints__(self, X, K, R, t, Kd):
        """
        Projects points X (3xN) using camera intrinsics K (3x3),
        extrinsics (R,t) and distortion parameters Kd=[k1,k2,p1,p2,k3].
        Roughly, x = K*(R*X + t) + distortion
        See http://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
        or cv2.projectPoints
        """

        x = (np.dot(R, X) + t)/ 100.0  # panoptic to kinect color scaling

        x[0:2, :] = x[0:2, :] / (x[2, :] + 1e-5)

        r = x[0, :] * x[0, :] + x[1, :] * x[1, :]

        # 去畸变
        x[0, :] = x[0, :] * (1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r
                            ) + 2 * Kd[2] * x[0, :] * x[1, :] + Kd[3] * (
                                r + 2 * x[0, :] * x[0, :])
        x[1, :] = x[1, :] * (1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r
                            ) + 2 * Kd[3] * x[0, :] * x[1, :] + Kd[2] * (
                                r + 2 * x[1, :] * x[1, :])

        x[0, :] = K[0, 0] * x[0, :] + K[0, 1] * x[1, :] + K[0, 2]
        x[1, :] = K[1, 0] * x[0, :] + K[1, 1] * x[1, :] + K[1, 2]

        return x


    def __get_affine_transform__(self, center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
        if isinstance(scale, torch.Tensor):
            scale = np.array(scale.cpu())
        if isinstance(center, torch.Tensor):
            center = np.array(center.cpu())
        if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
            scale = np.array([scale, scale])

        scale_tmp = scale * 200.0
        src_w, src_h = scale_tmp[0], scale_tmp[1]
        dst_w, dst_h = output_size[0], output_size[1]

        rot_rad = np.pi * rot / 180
        if src_w >= src_h:
            src_dir = self.__get_dir__([0, src_w * -0.5], rot_rad)
            dst_dir = np.array([0, dst_w * -0.5], np.float32)
        else:
            src_dir = self.__get_dir__([src_h * -0.5, 0], rot_rad)
            dst_dir = np.array([dst_h * -0.5, 0], np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale_tmp * shift     # x,y
        src[1, :] = center + src_dir + scale_tmp * shift
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

        src[2:, :] = self.__get_3rd_point__(src[0, :], src[1, :])
        dst[2:, :] = self.__get_3rd_point__(dst[0, :], dst[1, :])

        if inv:
            trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

        return trans

    def __get_dir__(self,src_point, rot_rad):
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)

        src_result = [0, 0]
        src_result[0] = src_point[0] * cs - src_point[1] * sn
        src_result[1] = src_point[0] * sn + src_point[1] * cs

        return src_result

    def __get_3rd_point__(self,a, b):
        direct = a - b
        return np.array(b) + np.array([-direct[1], direct[0]], dtype=np.float32)

    def __generate_target_heatmap__(self, joints, joints_vis):
        '''
        :param joints:  [[num_joints, 3]]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        nposes = len(joints)
        num_joints = self.num_joints
        target_weight = np.zeros((num_joints, 1), dtype=np.float32)
        for i in range(num_joints):
            for n in range(nposes):
                if joints_vis[n][i, 0] == 1:
                    target_weight[i, 0] = 1

        target = np.zeros(
            (num_joints, self.heatmap_size[1], self.heatmap_size[0]),
            dtype=np.float32)
        feat_stride = self.input_size / self.heatmap_size

        for n in range(nposes):
            human_scale = 2 * self.__compute_human_scale__(joints[n] / feat_stride, joints_vis[n]) # TODO: compute human scale
            if human_scale == 0:
                continue

            cur_sigma = self.sigma * np.sqrt((human_scale / (96.0 * 96.0)))
            tmp_size = cur_sigma * 3
            for joint_id in range(num_joints):
                feat_stride = self.input_size / self.heatmap_size
                mu_x = int(joints[n][joint_id][0] / feat_stride[0])
                mu_y = int(joints[n][joint_id][1] / feat_stride[1])
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if joints_vis[n][joint_id, 0] == 0 or \
                        ul[0] >= self.heatmap_size[0] or \
                        ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    continue

                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                g = np.exp(
                    -((x - x0)**2 + (y - y0)**2) / (2 * cur_sigma**2))

                # Usable gaussian range
                g_x = max(0,
                            -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0,
                            -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]],
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
            target = np.clip(target, 0, 1)

        return target, target_weight

    def __compute_human_scale__ (self, pose, joints_vis):
        idx = joints_vis[:, 0] == 1
        if np.sum(idx) == 0:
            return 0
        minx, maxx = np.min(pose[idx, 0]), np.max(pose[idx, 0])
        miny, maxy = np.min(pose[idx, 1]), np.max(pose[idx, 1])
        # return np.clip((maxy - miny) * (maxx - minx), 1.0 / 4 * 256**2,
        #                4 * 256**2)
        return np.clip(np.maximum(maxy - miny, maxx - minx)**2,  1.0 / 4 * 96**2, 4 * 96**2)    

if __name__ == '__main__':
    img_path = '/Extra/panzhiyu/CMU_kinect_data/'
    kp_path = '/Extra/panzhiyu/CMU_data/'
    view_set = [1,2,3]
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    depth_data = Panoptic_Depth(img_path, kp_path, view_set,is_train=True,
                                transform = transforms.Compose([
                                transforms.ToTensor(),
                                normalize,
                                ]))
    a,b,c,d,e,f,g,h,i = depth_data[1400]
    print('xx')

# 