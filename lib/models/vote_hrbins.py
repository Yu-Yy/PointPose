import torch
import torch.nn as nn
import torch.nn.functional as F

# from .miniViT import mViT
# from models import pose_higher_hrnet
# from models.depth_header import DepthEstimation
# from models.PM_header import PoseMaskEstimation
# from models.voteposenet import VotePoseNet
from models.hrnet_adabins import HrnetAdaptiveBins
from models.voteposenet import VotePoseNet

from depth_core.loss import SILogLoss, BinsChamferLoss, PerJointMSELoss, CrossEntropyMaskLoss, foreground_depth_loss
from depth_core.utils_depth import RunningAverage, compute_errors, RunningAverageDict, grad_generation, erode_generation, get_3d_points

from models.voteposenet import get_loss

class Vote_Hr_Adbins(nn.Module):
    def __init__(self, cfg, is_train = True, device=torch.device('cuda')):
        super(Vote_Hr_Adbins, self).__init__()
        self.cfg = cfg
        self.device = device
        self.hr_adbins = eval('HrnetAdaptiveBins.build')(cfg,is_train)
        self.num_proposals = 16
        self.votepose = VotePoseNet(cfg.MODEL_EXTRA.DECONV.NUM_CHANNELS[0], self.num_proposals)

    def forward(self, images, gt_depth, gt_valid_mask,gt_hm, gt_weights, gt_trans, gt_K, gt_M, gt_diff, gt_3d_pose):  # copy from the original function part
        criterion_hm = PerJointMSELoss().cuda()
        criterion_dense = SILogLoss().cuda()
        criterion_dense_attention = SILogLoss().cuda()
        criterion_chamfer = BinsChamferLoss().cuda()
        criterion_mask = CrossEntropyMaskLoss().cuda()
        criterion_fore_depth = foreground_depth_loss().cuda()
        generator_gradient = grad_generation().cuda()
        erode = erode_generation().cuda()

        orig_w = 1920
        orig_h = 1080
        points_extractor = get_3d_points(orig_W=orig_w, orig_H =orig_h).cuda()

        batch_num = images.shape[0]
        view_num = images.shape[1]
        number_joints = self.cfg.NETWORK.NUM_JOINTS

        loss_2d = torch.tensor(0).float().to(self.device) # set the initial 2d loss

        total_points = [[[] for _ in range(batch_num)] for _ in range(number_joints)]
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

            image = image.to(self.device)
            depth = depth.to(self.device)
            mask_gt = mask_gt.to(self.device)
            hm_gt = hm_gt.to(self.device)
            wt_gt = wt_gt.to(self.device)
            trans_matrix = trans_matrix.to(self.device)
            K_matrix = K_matrix.to(self.device)
            M_matrix = M_matrix.to(self.device)
            diff = diff.to(self.device)

            bin_edges, pred, heatmap, mask_prob, feature_out = self.hr_adbins(image)

            # depth loss
            depth_mask = depth > self.cfg.DATASET.MIN_DEPTH
            attention_mask = depth_mask * mask_gt   
            l_dense = criterion_dense(pred, depth, mask=depth_mask.to(torch.bool), interpolate=True)
            l_dense_attention = criterion_dense_attention(pred, depth, mask=attention_mask.to(torch.bool), interpolate=True)
            l_chamfer = criterion_chamfer(bin_edges, depth.float())

            l_fore_depth =criterion_fore_depth(pred, depth, mask=attention_mask.to(torch.bool), interpolate=True)
            loss_depth = l_dense + 0.1 * l_chamfer +  l_dense_attention 
            
            # hm loss
            loss_hm = criterion_hm(heatmap, hm_gt, mask_gt ,True, wt_gt)
            
             # mask loss
            loss_mask = criterion_mask(mask_prob, mask_gt.long())

            loss_2d = loss_2d + loss_depth + 50 * loss_hm + loss_mask * 5  

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
            filter_mask = []
            for hm_idx in range(number_joints):
                hm_sampling = heatmap[:,hm_idx:hm_idx + 1,:,:]
                hm_max = torch.max(hm_sampling)
                choice_generator = 2 * hm_max * torch.rand(hm_sampling.shape).to(self.device)
                hm_sampling_mask = hm_sampling > choice_generator
                filter_mask.append(erode_mask * filter_gradient_mask * hm_sampling_mask)
            total_points = points_extractor(pred_process, filter_mask, K_matrix, M_matrix,diff, trans_matrix, total_points, feature_out.clone())

        for hm_idx in range(number_joints):
            for b in range(batch_num):
                total_points[hm_idx][b] = torch.cat(total_points[hm_idx][b],dim=0)
        # existing the 0 joints sampling
        ###### get in the pointcloud input mode
        for hm_idx in range(number_joints):
            batch_total_points = total_points[hm_idx]
            max_points_num = 0
            for b in range(batch_num):
                s_batch_points = batch_total_points[b]
                s_num = s_batch_points.shape[0]
                if s_num >= max_points_num:
                    max_points_num = s_num
            # assure the max_points_num
            for b in range(batch_num):
                s_batch_points = batch_total_points[b]
                s_num = s_batch_points.shape[0]
                if s_num == max_points_num:
                    total_points[hm_idx][b] = s_batch_points.unsqueeze(0)
                    continue
                offset = max_points_num - s_num
                if s_num == 0:  # 其他batch 有值，该batch 在该关节点从处无值
                    fill_tensor = torch.zeros((max_points_num, 35)).to(self.device)
                else:
                    fill_indx = torch.randint(s_num, (offset, ))
                    fill_tensor = s_batch_points[fill_indx,:] # 索引可能不够了
                new_tensor = torch.cat([s_batch_points,fill_tensor],dim=0) # points must be float tensor
                total_points[hm_idx][b] = new_tensor.unsqueeze(0)

            total_points[hm_idx] = torch.cat(total_points[hm_idx],dim=0)
        
        # send into the vote net 
        # loss_3d = torch.tensor(0).float().to(self.device)
        loss_distance = torch.tensor(0).float().to(self.device)
        loss_objective = torch.tensor(0).float().to(self.device)
        loss_vote = torch.tensor(0).float().to(self.device)
        predicted_3dpose = []
        for hm_idx in range(number_joints):
            if total_points[hm_idx].shape[1] == 0:
                predicted_3dpose.append(torch.zeros((batch_num, self.num_proposals,3)).to(self.device))
                continue
            # import pdb; pdb.set_trace()
            end_points = self.votepose(total_points[hm_idx])
            # generate the corresponding gt_points

            gt_points = gt_3d_pose[:,:,hm_idx,:3].float() / 100  # gt_3d_pose B,N,J,4  for metric
            loss, end_points = get_loss(end_points, gt_points) # B, valid_num,3
            loss_distance = loss_distance + end_points['distance_loss']
            loss_objective = loss_objective + end_points['objectness_loss']
            loss_vote = loss_vote + end_points['vote_loss']

            predicted_3dpose.append(end_points['center'])
            # loss_3d = loss_3d + loss
        
        # print('forward OK!')
        # print(f"{loss_3d} loss 3d")
        # print(loss_3d)
        # print(f"{loss_2d} loss 2d")
        # print(loss_2d)
        # print(predicted_3dpose)
        return loss_vote,loss_objective,loss_distance,loss_2d, predicted_3dpose















