import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt

# from .miniViT import mViT
# from models import pose_higher_hrnet
# from models.depth_header import DepthEstimation
# from models.PM_header import PoseMaskEstimation
# from models.voteposenet import VotePoseNet
from models.hrnet_adabins import HrnetAdaptiveBins
from models.voteposenet import VotePoseNet,VotePoseNet_
# import models
from depth_core.loss import SILogLoss, BinsChamferLoss, PerJointMSELoss, CrossEntropyMaskLoss, foreground_depth_loss
from depth_core.utils_depth import RunningAverage, compute_errors, RunningAverageDict, grad_generation, erode_generation, get_3d_points

from models.voteposenet import get_loss, get_loss_

class Vote_Hr_Adbins(nn.Module):
    def __init__(self, cfg, is_train = True, num_max_points = 16384, device=torch.device('cuda')):
        super(Vote_Hr_Adbins, self).__init__()
        self.cfg = cfg
        self.device = device
        self.hr_adbins =  eval('HrnetAdaptiveBins.build')(cfg,is_train) #'models.'+ cfg.MODEL+'.build'
        self.num_proposals = 128
        self.num_max_points = num_max_points
        self.votepose = VotePoseNet_(cfg.MODEL_EXTRA.DECONV.NUM_CHANNELS[0], self.num_proposals, vote_factor=cfg.NETWORK.NUM_JOINTS)

    def forward(self, images, gt_depth, gt_valid_mask,gt_hm, gt_paf, gt_weights, gt_trans, gt_K, gt_M, gt_diff, gt_3d_pose, epoch=0, idx = 0):  # copy from the original function part
                        
        criterion_hm = PerJointMSELoss().cuda()
        criterion_dense = SILogLoss().cuda()
        criterion_dense_attention = SILogLoss().cuda()
        criterion_chamfer = BinsChamferLoss().cuda()
        criterion_mask = CrossEntropyMaskLoss().cuda()
        criterion_paf = PerJointMSELoss().cuda()
        # criterion_fore_depth = foreground_depth_loss().cuda()
        generator_gradient = grad_generation().cuda()
        erode = erode_generation().cuda()

        orig_w = 1920
        orig_h = 1080
        points_extractor = get_3d_points(orig_W=orig_w, orig_H =orig_h).cuda()

        batch_num = images.shape[0]
        view_num = images.shape[1]
        number_joints = self.cfg.NETWORK.NUM_JOINTS

        loss_2d = torch.tensor(0).float().to(self.device) # set the initial 2d loss
        loss_depth = torch.tensor(0).float().to(self.device)
        loss_hm = torch.tensor(0).float().to(self.device)
        loss_mask = torch.tensor(0).float().to(self.device)
        loss_paf = torch.tensor(0).float().to(self.device)
        total_points = [[] for _ in range(batch_num)]
        total_vectors = [[] for _ in range(batch_num)]
        for view in range(view_num):
            image = images[:,view,...]
            depth = gt_depth[:,view,...]
            mask_gt = gt_valid_mask[:,view,...]
            hm_gt = gt_hm[:,view,...]
            wt_gt = gt_weights[:,view,...]
            trans_matrix = gt_trans[:,view,...]
            K_matrix = gt_K[:,view,...]
            M_matrix = gt_M[:,view,...]
            diff = gt_diff[:,view,...]
            paf = gt_paf[:,view,...]

            image = image.to(self.device)
            depth = depth.to(self.device)
            mask_gt = mask_gt.to(self.device)
            hm_gt = hm_gt.to(self.device)
            wt_gt = wt_gt.to(self.device)
            trans_matrix = trans_matrix.to(self.device)
            K_matrix = K_matrix.to(self.device)
            M_matrix = M_matrix.to(self.device)
            diff = diff.to(self.device)
            paf = paf.to(self.device)
            # import pdb;pdb.set_trace()
            bin_edges, pred, depth_probability, heatmap, paf_pred ,mask_prob, feature_out = self.hr_adbins(image)
            # import pdb;pdb.set_trace()
            # depth_probability = torch.max(depth_probability, dim=1, keepdim=True)[0] # return the number

            # plot the relationship between prob and error
            # fitted_depth = F.interpolate(depth, scale_factor=0.5)
            # error_test = torch.pow((pred - fitted_depth),2)
            # show_prob = depth_probability[0,...].cpu().numpy()
            # show_error = error_test[0,...].cpu().numpy()
            # show_prob = show_prob.reshape(-1)
            # show_error = show_error.reshape(-1)
            
            # plotting_relations = plt.figure()
            # plt.scatter(show_prob,show_error)
            # plt.savefig('test_relation.png')
            # plt.cla()
            # plt.close(plotting_relations)
            # import pdb;pdb.set_trace()

            # depth loss
            depth_mask = depth > self.cfg.DATASET.MIN_DEPTH
            attention_mask = depth_mask * mask_gt   
            l_dense = criterion_dense(pred, depth, mask=depth_mask.to(torch.bool), interpolate=True)
            l_dense_attention = criterion_dense_attention(pred, depth, mask=attention_mask.to(torch.bool), interpolate=True)
            l_chamfer = criterion_chamfer(bin_edges, depth.float())

            # l_fore_depth =criterion_fore_depth(pred, depth, mask=attention_mask.to(torch.bool), interpolate=True)
            l_depth = l_dense + l_dense_attention  + 0.1 * l_chamfer 
            
            # hm loss
            l_hm = criterion_hm(heatmap, hm_gt, mask_gt ,True, wt_gt)
            # paf loss 
            l_paf = criterion_paf(paf_pred, paf, mask_gt, False)
             # mask loss
            l_mask = criterion_mask(mask_prob, mask_gt.long())

            loss_2d = loss_2d + l_depth + 50 * l_hm + l_mask * 5 + l_paf * 50
            loss_depth = loss_depth + l_depth
            loss_hm = loss_hm + 50 * l_hm
            loss_mask = loss_mask + l_mask * 5
            loss_paf = loss_paf + l_paf * 50

            # if  (epoch>=1) or (idx > 3000): 
                # for epoch 0, do not consider the 3d part
                # 2d network can be trained by the loss loss_2d
            pred_process = pred.clone() # for projection # TODO: avoid backward
            
            # 1. get the mask to do the filtering as "number"
            fitted_mask = F.interpolate(mask_prob, scale_factor=0.5)
            # filter mask
            filter_gradient_mask = generator_gradient(pred_process.clone())

            # convolution is 2*P = F_size -1 for same 
            kernel = torch.ones(5,5).to(self.device)  
            mask = (fitted_mask > 0.5).int()
            # erosion mask
            erode_mask = erode(mask,kernel.detach()) # erode error
            # one frame mask
            # filter_mask = []
            # for hm_idx in range(number_joints):
            hm_sampling,_ = torch.max(heatmap, dim = 1, keepdims=True)
            hm_max = torch.max(hm_sampling) # 
            choice_generator = hm_max * torch.rand(hm_sampling.shape).to(self.device) # orig rand 
            hm_sampling_mask = hm_sampling > choice_generator
            filter_mask = erode_mask * filter_gradient_mask  #* hm_sampling_mask #
            # judge the valid region of the mask
            # B,C,H,W = filter_mask.shape # filter mask 的回归质量
            # if torch.sum(filter_mask) < 0.01 * H * W * B * C:
            #     continue
            
            total_points, total_vectors = points_extractor(pred_process, filter_mask, K_matrix, M_matrix, diff, trans_matrix, total_points, total_vectors, feature_out.clone())

        # for hm_idx in range(number_joints):
        # judge the valid of the total_points
        # valid = torch.tensor(True).to(self.device)
        # for b in range(batch_num):
        #     if len(total_points[b]) == 0:
        #         valid = ~valid
        #         break
        # if ((epoch>=1) or (idx > 3000)) and valid: 
        for b in range(batch_num):
            total_points[b] = torch.cat(total_points[b],dim=0) # view fusion
            total_vectors[b] = torch.cat(total_vectors[b],dim=0) # ray vector (no direction distinguish)
        # import pdb;pdb.set_trace()
        # existing the 0 joints sampling
        ###### get in the pointcloud input mode
        # for hm_idx in range(number_joints):
        batch_total_points = total_points.copy()
        batch_total_vectors = total_vectors.copy()
        max_points_num = 0
        for b in range(batch_num):
            s_batch_points = batch_total_points[b]
            s_num = s_batch_points.shape[0]
            if s_num >= max_points_num:
                max_points_num = s_num
        
        if max_points_num > self.num_max_points:
            for b in range(batch_num):
                s_batch_points = batch_total_points[b]
                s_batch_vectors = batch_total_vectors[b]
                s_num = s_batch_points.shape[0]
                if s_num <= self.num_max_points:
                    offset = self.num_max_points - s_num
                    if s_num == 0:  # 其他batch 有值，该batch 在该关节点从处无值
                        fill_tensor = torch.zeros((max_points_num, 35)).to(self.device)
                        fill_vectors = torch.zeros((max_points_num, 3)).to(self.device)
                    else:
                        fill_indx = torch.randint(s_num, (offset, ))
                        fill_tensor = s_batch_points[fill_indx,:] # 索引可能不够了
                        fill_vectors = s_batch_vectors[fill_indx,:]
                    new_tensor = torch.cat([s_batch_points,fill_tensor],dim=0) # points must be float tensor
                    new_vector = torch.cat([s_batch_vectors,fill_vectors],dim=0)
                    total_points[b] = new_tensor.unsqueeze(0)
                    total_vectors[b] = new_vector.unsqueeze(0)
                else:
                    sample_idx = torch.randperm(s_num)[:self.num_max_points]
                    new_tensor = s_batch_points[sample_idx,:]
                    total_vectors[b] = new_vector.unsqueeze(0)
                    total_points[b] = new_tensor.unsqueeze(0)
        else:
            for b in range(batch_num):
                s_batch_points = batch_total_points[b]
                s_batch_vectors = batch_total_vectors[b]
                s_num = s_batch_points.shape[0]
                if s_num == max_points_num:
                    total_vectors[b]
                    total_points[b] = s_batch_points.unsqueeze(0)
                    total_vectors[b] = s_batch_vectors.unsqueeze(0)
                    continue
                offset = max_points_num - s_num
                if s_num == 0:  # 其他batch 有值，该batch 在该关节点从处无值
                    fill_tensor = torch.zeros((max_points_num, 35)).to(self.device)
                    fill_vectors = torch.zeros((max_points_num, 3)).to(self.device)
                else:
                    fill_indx = torch.randint(s_num, (offset, ))
                    fill_tensor = s_batch_points[fill_indx,:] # 索引可能不够了
                    fill_vectors = s_batch_vectors[fill_indx,:]
                new_tensor = torch.cat([s_batch_points,fill_tensor],dim=0) # points must be float tensor
                new_vector = torch.cat([s_batch_vectors,fill_vectors],dim=0)
                total_points[b] = new_tensor.unsqueeze(0)
                total_vectors[b] = new_vector.unsqueeze(0)

        total_points = torch.cat(total_points,dim=0)
        total_vectors = torch.cat(total_vectors,dim=0)
        # send into the vote net 
        # loss_3d = torch.tensor(0).float().to(self.device)
        # loss_distance = torch.tensor(0).float().to(self.device)
        # loss_objective = torch.tensor(0).float().to(self.device)
        # loss_vote = torch.tensor(0).float().to(self.device)
        # predicted_3dpose = []
        end_points = self.votepose(total_points)
            # generate the corresponding gt_points

        gt_points = gt_3d_pose[:,:,:,:3].float() / 100  # gt_3d_pose B,N,J,4  for metric
        loss_3d, end_points = get_loss_(end_points, gt_points) # B, valid_num,3
        loss_distance = end_points['distance_loss']
        loss_objective = end_points['objectness_loss']
        loss_vote =  end_points['vote_loss']

        predicted_3dpose = end_points['center_list'] # It is should be the dict
        # decode its joints into B * N * P * 3
        center = []
        for j_ in range(number_joints):
            center.append(predicted_3dpose[j_]['center'].unsqueeze(1))
        center = torch.cat(center,dim=1)

        vote_coord = end_points['vote_xyz'] # TODO: vis by debug
            # get its center
        # else:
        #     print('warning the no_grad')   #
        #     loss_vote = torch.tensor(0.).to(self.device)
        #     loss_objective = torch.tensor(0.).to(self.device)
        #     loss_distance = torch.tensor(0.).to(self.device)
        #     loss_3d = torch.tensor(0.).to(self.device)
        #     center = torch.zeros((B,number_joints,self.num_proposals,3)).to(self.device)
            # predicted_3dpose = torch.tensor(0.).to(self.device)
            

        # for hm_idx in range(number_joints):
        #     if total_points[hm_idx].shape[1] == 0:
        #         predicted_3dpose.append(torch.zeros((batch_num, self.num_proposals,3)).to(self.device))
        #         continue
        #     # import pdb; pdb.set_trace()
        #     end_points = self.votepose(total_points[hm_idx])
        #     # generate the corresponding gt_points

        #     gt_points = gt_3d_pose[:,:,hm_idx,:3].float() / 100  # gt_3d_pose B,N,J,4  for metric
        #     loss, end_points = get_loss(end_points, gt_points) # B, valid_num,3
        #     loss_distance = loss_distance + end_points['distance_loss']
        #     loss_objective = loss_objective + end_points['objectness_loss']
        #     loss_vote = loss_vote + end_points['vote_loss']

        #     predicted_3dpose.append(end_points['center'])
            # loss_3d = loss_3d + loss
        
        # print('forward OK!')
        # print(f"{loss_3d} loss 3d")
        # print(loss_3d)
        # print(f"{loss_2d} loss 2d")
        # print(loss_2d)
        # print(predicted_3dpose)
        # import pdb; pdb.set_trace()
        return loss_vote, loss_objective, loss_distance, loss_3d,loss_depth, loss_hm, loss_paf,loss_mask ,loss_2d, center, pred, heatmap, mask_prob, vote_coord     #, valid















