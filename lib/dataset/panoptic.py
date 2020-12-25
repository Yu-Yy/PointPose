# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os.path as osp
import numpy as np
import json_tricks as json
import pickle
import logging
import os 
import copy

from dataset.JointsDataset import JointsDataset
from utils.transforms import projectPoints

logger = logging.getLogger(__name__)

TRAIN_LIST = [
    '160422_ultimatum1',
    '160224_haggling1',
    '160226_haggling1',
    '161202_haggling1',
    '160906_ian1',
    '160906_ian2',
    '160906_ian3',
    '160906_band1',
    '160906_band2',
    '160906_band3',
]
VAL_LIST = ['160906_pizza1', '160422_haggling1', '160906_ian5', '160906_band4']
# VAL_LIST = ['160226_haggling1new', '160906_pizza1new', '160906_band4new','160906_ian5new']
# VAL_LIST = ['171026_pose3']


coco_joints_def = {0: 'nose',
                   1: 'Leye', 2: 'Reye', 3: 'Lear', 4: 'Rear',
                   5: 'Lsho', 6: 'Rsho',
                   7: 'Lelb', 8: 'Relb',
                   9: 'Lwri', 10: 'Rwri',
                   11: 'Lhip', 12: 'Rhip',
                   13: 'Lkne', 14: 'Rkne',
                   15: 'Lank', 16: 'Rank'}

LIMBS = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [11, 13], [13, 15],
        [6, 12], [12, 14], [14, 16], [5, 6], [11, 12]]

# changed to COCO standard
JOINTS_DEF = {
    'neck': 0,
    'nose': 1,
    'mid-hip': 2,
    'l-shoulder': 3,
    'l-elbow': 4,
    'l-wrist': 5,
    'l-hip': 6,
    'l-knee': 7,
    'l-ankle': 8,
    'r-shoulder': 9,
    'r-elbow': 10,
    'r-wrist': 11,
    'r-hip': 12,
    'r-knee': 13,
    'r-ankle': 14,
    # 'l-eye': 15,
    # 'l-ear': 16,
    # 'r-eye': 17,
    # 'r-ear': 18, # 恢复后四点
} # panoptic 实际使用关节点数为15

# change the original annotation to the COCo19 

# cmu2coco [1,15,17,16,18,3,9,4,10,5,11,6,12,7,13,8,14] 

# original CMU limb defination
# LIMBS = [[0, 1],
#          [0, 2],
#          [0, 3],
#          [3, 4],
#          [4, 5],
#          [0, 9],
#          [9, 10],
#          [10, 11],
#          [2, 6],
#          [2, 12],
#          [6, 7],
#          [7, 8],
#          [12, 13],
#          [13, 14]]

# coco limb defination
# LIMBS = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [11, 13], [13, 15],
#         [6, 12], [12, 14], [14, 16], [5, 6], [11, 12]]


class Panoptic(JointsDataset):
    def __init__(self, cfg, image_set, is_train, transform=None):
        super().__init__(cfg, image_set, is_train, transform)
        self.pixel_std = 200.0
        self.joints_def = coco_joints_def
        self.limbs = LIMBS
        self.num_joints = len(coco_joints_def) # changed to coco standard#

        if self.image_set == 'train':
            self.sequence_list = TRAIN_LIST
            self._interval = 3
            self.cam_list = [(0, 12), (0, 6), (0, 23), (0, 13), (0, 3)][:self.num_views]
            # self.cam_list = list(set([(0, n) for n in range(0, 31)]) - {(0, 12), (0, 6), (0, 23), (0, 13), (0, 3)})
            # self.cam_list.sort()
            self.num_views = len(self.cam_list)
        elif self.image_set == 'validation':
            self.sequence_list = VAL_LIST
            self._interval = 3 #12
            self.cam_list = [(0, 12), (0, 6), (0, 23), (0, 13), (0, 3),(0, 15), (0, 7), (0, 20), (0, 5), (0, 1)][:self.num_views]
            # self.cam_list = [(0, 15), (0, 7), (0, 20), (0, 5), (0, 1)][:self.num_views] # 1 5 7 15 20
            self.num_views = len(self.cam_list)

        self.db_file = 'group_{}_cam{}_f17_cmucoco.pkl'.format(self.image_set, self.num_views)   # f17_cmucoco_testd5
        self.dataset_root = '/Extra/panzhiyu/CMU_data/' # change the diractory
        self.db_file = os.path.join(self.dataset_root, self.db_file)

        if osp.exists(self.db_file):
            info = pickle.load(open(self.db_file, 'rb'))
            assert info['sequence_list'] == self.sequence_list
            assert info['interval'] == self._interval
            assert info['cam_list'] == self.cam_list
            self.db = info['db'] # 基本参数和输入
        else:
            self.db = self._get_db()
            info = {
                'sequence_list': self.sequence_list,
                'interval': self._interval,
                'cam_list': self.cam_list,
                'db': self.db
            }
            pickle.dump(info, open(self.db_file, 'wb'))
        # self.db = self._get_db()
        self.db_size = len(self.db) # db number

    def _get_db(self):
        width = 1920
        height = 1080
        db = []
        for seq in self.sequence_list:

            cameras = self._get_cam(seq)

            curr_anno = osp.join(self.dataset_root, seq, 'hdPose3d_stage1_coco19') 
            anno_files = sorted(glob.iglob('{:s}/*.json'.format(curr_anno))) # 姿态的标注文件

            for i, file in enumerate(anno_files):
                if i % self._interval == 0:  # 进行选择，间隔下提取真值，进行测试，并不是全部测试
                    with open(file) as dfile:
                        bodies = json.load(dfile)['bodies']
                    if len(bodies) == 0: #当前标注无人
                        continue

                    for k, v in cameras.items(): # 视角及参数 遍历各个视角 pose reading 有冗余成分
                        postfix = osp.basename(file).replace('body3DScene', '')
                        prefix = '{:02d}_{:02d}'.format(k[0], k[1])
                        image = osp.join(seq, 'hdImgs', prefix,
                                         prefix + postfix) # 看命名规则
                        image = image.replace('json', 'jpg') #找到对应标注的视角图片

                        all_poses_3d = []
                        all_poses_vis_3d = []
                        all_poses = []
                        all_poses_vis = []
                        for body in bodies:
                            pose3d = np.array(body['joints19']).reshape((-1, 4)) # reshape operation
                            
                            # pose3d = pose3d[:self.num_joints] #筛选前15个 original

                            pose3d_coco = np.zeros([self.num_joints,4])
                            pose3d_coco[:] = pose3d[[1,15,17,16,18,3,9,4,10,5,11,6,12,7,13,8,14],:]
                            pose3d = pose3d_coco.copy() # new pose3d in coco standard 
                            
                            

                            joints_vis = pose3d[:, -1] > 0.1 # decide the visibility according to the joints 返回bool 值

                            # print(self.root_id)
                            if len(self.root_id) == 1:
                                if not joints_vis[self.root_id]: # Midhip 不可被挡, 否则直接抛弃
                                    continue
                            else:
                                if (not joints_vis[self.root_id[0]]) and (not joints_vis[self.root_id[1]]): # Midhip 不可被挡, 否则直接抛弃
                                    continue

                            # Coordinate transformation
                            M = np.array([[1.0, 0.0, 0.0],
                                          [0.0, 0.0, -1.0],
                                          [0.0, 1.0, 0.0]])
                            pose3d[:, 0:3] = pose3d[:, 0:3].dot(M) # pose处理，（X,-Z,Y）

                            all_poses_3d.append(pose3d[:, 0:3] * 10.0) # *10 process ? 15*3
                            all_poses_vis_3d.append(
                                np.repeat(
                                    np.reshape(joints_vis, (-1, 1)), 3, axis=1)) # 15*3

                            pose2d = np.zeros((pose3d.shape[0], 2))
                            pose2d[:, :2] = projectPoints(
                                pose3d[:, 0:3].transpose(), v['K'], v['R'],
                                v['t'], v['distCoef']).transpose()[:, :2]   # 投影到2D 平面
                            x_check = np.bitwise_and(pose2d[:, 0] >= 0, 
                                                     pose2d[:, 0] <= width - 1) #(15,) bool
                            y_check = np.bitwise_and(pose2d[:, 1] >= 0,
                                                     pose2d[:, 1] <= height - 1)
                            check = np.bitwise_and(x_check, y_check) # check bool se
                            joints_vis[np.logical_not(check)] = 0 # 2D visible 第二判据, 在

                            
                            all_poses.append(pose2d) # 2D 
                            all_poses_vis.append(
                                np.repeat(
                                    np.reshape(joints_vis, (-1, 1)), 2, axis=1)) # visble *x  标签跟pose同维度
                            



                        

                        if len(all_poses_3d) > 0:
                            our_cam = {}
                            our_cam['R'] = v['R']
                            our_cam['T'] = -np.dot(v['R'].T, v['t']) * 10.0  # cm to mm
                            our_cam['fx'] = np.array(v['K'][0, 0])
                            our_cam['fy'] = np.array(v['K'][1, 1])
                            our_cam['cx'] = np.array(v['K'][0, 2])
                            our_cam['cy'] = np.array(v['K'][1, 2])
                            our_cam['k'] = v['distCoef'][[0, 1, 4]].reshape(3, 1)
                            our_cam['p'] = v['distCoef'][[2, 3]].reshape(2, 1)

                            db.append({
                                'key': "{}_{}{}".format(seq, prefix, postfix.split('.')[0]),
                                'image': osp.join(self.dataset_root, image),
                                'joints_3d': all_poses_3d,
                                'joints_3d_vis': all_poses_vis_3d,
                                'joints_2d': all_poses,
                                'joints_2d_vis': all_poses_vis,
                                'camera': our_cam
                            })
        return db

    def _get_cam(self, seq):
        cam_file = osp.join(self.dataset_root, seq, 'calibration_{:s}.json'.format(seq)) 
        with open(cam_file) as cfile:
            calib = json.load(cfile)

        M = np.array([[1.0, 0.0, 0.0],
                      [0.0, 0.0, -1.0],
                      [0.0, 1.0, 0.0]])
        cameras = {}
        for cam in calib['cameras']:
            if (cam['panel'], cam['node']) in self.cam_list: # camera 位置信息的选择 （panel, node） 当前，视角就是Node决定
                sel_cam = {}
                sel_cam['K'] = np.array(cam['K'])
                sel_cam['distCoef'] = np.array(cam['distCoef'])
                sel_cam['R'] = np.array(cam['R']).dot(M)  # 旋转矩阵要处理一下 （坐标设置跟投影矩阵不匹配？）
                sel_cam['t'] = np.array(cam['t']).reshape((3, 1))
                cameras[(cam['panel'], cam['node'])] = sel_cam
        return cameras

    def __getitem__(self, idx):
        input, target, weight, target_3d, meta, input_heatmap = [], [], [], [], [], []

        # if self.image_set == 'train':
        #     # camera_num = np.random.choice([5], size=1)
        #     select_cam = np.random.choice(self.num_views, size=5, replace=False)
        # elif self.image_set == 'validation':
        #     select_cam = list(range(self.num_views))

        for k in range(self.num_views): # 输出不同视角的一个场景
            i, t, w, t3, m, ih = super().__getitem__(self.num_views * idx + k) # 父类读图
            # ！！original
            if i is None: # 保证有图像的输入
                continue # 
            # 先保证input 不为空
            input.append(i)
            target.append(t)
            weight.append(w)
            target_3d.append(t3)
            meta.append(m)
            input_heatmap.append(ih)
        return input, target, weight, target_3d, meta, input_heatmap

    def __len__(self):
        return self.db_size // self.num_views # // 整除 一次取num_views 个视角 # db_size # of pictures in one view

    def evaluate(self, preds):
        eval_list = []
        gt_num = self.db_size // self.num_views
        assert len(preds) == gt_num, 'number mismatch'

        total_gt = 0
        for i in range(gt_num):
            index = self.num_views * i # evaluate 只进行3D的比较 
            db_rec = copy.deepcopy(self.db[index])
            joints_3d = db_rec['joints_3d']
            joints_3d_vis = db_rec['joints_3d_vis']

            if len(joints_3d) == 0:
                continue

            pred = preds[i].copy()
            pred = pred[pred[:, 0, 3] >= 0]
            for pose in pred:
                mpjpes = []
                for (gt, gt_vis) in zip(joints_3d, joints_3d_vis):
                    vis = gt_vis[:, 0] > 0
                    mpjpe = np.mean(np.sqrt(np.sum((pose[vis, 0:3] - gt[vis]) ** 2, axis=-1)))
                    mpjpes.append(mpjpe)
                min_gt = np.argmin(mpjpes)
                min_mpjpe = np.min(mpjpes)
                score = pose[0, 4]
                eval_list.append({
                    "mpjpe": float(min_mpjpe),
                    "score": float(score),
                    "gt_id": int(total_gt + min_gt)
                })

            total_gt += len(joints_3d)

        mpjpe_threshold = np.arange(25, 155, 25)
        aps = []
        recs = []
        for t in mpjpe_threshold:
            ap, rec = self._eval_list_to_ap(eval_list, total_gt, t)
            aps.append(ap)
            recs.append(rec)

        return aps, recs, self._eval_list_to_mpjpe(eval_list), self._eval_list_to_recall(eval_list, total_gt)

    @staticmethod
    def _eval_list_to_ap(eval_list, total_gt, threshold):
        eval_list.sort(key=lambda k: k["score"], reverse=True)
        total_num = len(eval_list)

        tp = np.zeros(total_num)
        fp = np.zeros(total_num)
        gt_det = []
        for i, item in enumerate(eval_list):
            if item["mpjpe"] < threshold and item["gt_id"] not in gt_det:
                tp[i] = 1
                gt_det.append(item["gt_id"])
            else:
                fp[i] = 1
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        recall = tp / (total_gt + 1e-5)
        precise = tp / (tp + fp + 1e-5)
        for n in range(total_num - 2, -1, -1):
            precise[n] = max(precise[n], precise[n + 1])

        precise = np.concatenate(([0], precise, [0]))
        recall = np.concatenate(([0], recall, [1]))
        index = np.where(recall[1:] != recall[:-1])[0]
        ap = np.sum((recall[index + 1] - recall[index]) * precise[index + 1])

        return ap, recall[-2]

    @staticmethod
    def _eval_list_to_mpjpe(eval_list, threshold=500):
        eval_list.sort(key=lambda k: k["score"], reverse=True)
        gt_det = []

        mpjpes = []
        for i, item in enumerate(eval_list):
            if item["mpjpe"] < threshold and item["gt_id"] not in gt_det:
                mpjpes.append(item["mpjpe"])
                gt_det.append(item["gt_id"])

        return np.mean(mpjpes) if len(mpjpes) > 0 else np.inf

    @staticmethod
    def _eval_list_to_recall(eval_list, total_gt, threshold=500):
        gt_ids = [e["gt_id"] for e in eval_list if e["mpjpe"] < threshold]

        return len(np.unique(gt_ids)) / total_gt




