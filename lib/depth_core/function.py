from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os
import copy
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.functional import interpolate
from depth_core.loss import SILogLoss, BinsChamferLoss, PerJointMSELoss, CrossEntropyMaskLoss, SILogLoss_grad, foreground_depth_loss
from depth_core.utils_depth import RunningAverage, compute_errors, RunningAverageDict, grad_generation, erode_generation, get_3d_points
# define the connection
CONNS =  [[0, 1],
         [0, 2],
         [0, 3],
         [3, 4],
         [4, 5],
         [0, 9],
         [9, 10],
         [10, 11],
         [2, 6],
         [2, 12],
         [6, 7],
         [7, 8],
         [12, 13],
         [13, 14]]
# from models.hrnet_adabins import HrnetAdaptiveBins
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import open3d as o3d
import cv2

logger = logging.getLogger(__name__)


def train_depth(config, model, optimizer, loader, epoch, output_dir, writer_dict, device=torch.device('cuda'), dtype=torch.float, logger=logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_depth = AverageMeter()
    losses_hm = AverageMeter()
    losses_mask = AverageMeter()
    losses_fore_depth = AverageMeter()
    losses_paf = AverageMeter()
    
    criterion_hm = PerJointMSELoss().cuda()
    criterion_paf = PerJointMSELoss().cuda()
    criterion_dense = SILogLoss().cuda()
    criterion_dense_attention = SILogLoss().cuda()
    criterion_chamfer = BinsChamferLoss().cuda()
    criterion_mask = CrossEntropyMaskLoss().cuda()
    # criterion_grad = SILogLoss_grad().cuda()
    criterion_fore_depth = foreground_depth_loss().cuda()

    model.train()
    
    number_joints = config.NETWORK.NUM_JOINTS

    end = time.time()
    # multi_loader_training shelf and cmu
    for i, batch in enumerate(loader): # 
        if len(batch) == 0:
            continue
        image, depth, mask_gt, hm_gt, wt_gt, paf_gt = batch
        image = image.to(device)
        depth = depth.to(device)
        mask_gt = mask_gt.to(device)
        hm_gt = hm_gt.to(device)
        wt_gt = wt_gt.to(device)
        paf_gt = paf_gt.to(device)
        
        # import pdb; pdb.set_trace()
        bin_edges, pred,heatmap, paf_pred, mask_prob, feature_out = model(image) # depth is in half resolution depth_uncertainty ,

        # depth loss
        depth_mask = depth > config.DATASET.MIN_DEPTH
        attention_mask = depth_mask * mask_gt
        l_dense = criterion_dense(pred, depth, mask=depth_mask.to(torch.bool), interpolate=True) #depth_uncertainty,
        # add one heatmap mask to pay more attention to the keypoint position


        l_dense_attention = criterion_dense_attention(pred,depth, mask=attention_mask.to(torch.bool), interpolate=True)# depth_uncertainty,
        l_chamfer = criterion_chamfer(bin_edges, depth.float())
        # l_grad = criterion_grad(pred, depth.detach().float(), mask=attention_mask.to(torch.bool), interpolate=True) # do not add this loss 
        # print(l_grad)

        l_fore_depth =criterion_fore_depth(pred, depth, mask=attention_mask.to(torch.bool), interpolate=True)
        losses_fore_depth.update(l_fore_depth.item())

        loss_depth = l_dense + 0.1 * l_chamfer +  l_dense_attention # pay more attention to the  #+ 2e-3 * l_grad # pay attention to the foreground and control the grad
        losses_depth.update(loss_depth.item())  # attention avaliable (not to be no response)

        # hm loss
        # loss_hm = criterion_hm(heatmap, hm_gt, mask_gt ,True, wt_gt)
        loss_hm = criterion_hm(heatmap, hm_gt, mask_gt, False) # no vis label
        losses_hm.update(loss_hm.item() * 50)
        # paf loss 
        loss_paf = criterion_paf(paf_pred, paf_gt, mask_gt, False)
        losses_paf.update(loss_paf.item() * 50)
        # mask loss
        loss_mask = criterion_mask(mask_prob, mask_gt.long())
        losses_mask.update(loss_mask.item() * 5)

        loss = loss_depth + 50 * loss_hm + loss_mask * 5 + loss_paf * 50
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward() # add loss 3d 
        optimizer.step()

        data_time.update(time.time() - end)
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            gpu_memory_usage = torch.cuda.memory_allocated(0)
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed: {speed:.1f} samples/s\t' \
                  'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss: {loss.val:.6f} ({loss.avg:.6f})\t' \
                  'Loss_depth: {loss_2d.val:.7f} ({loss_2d.avg:.7f})\t' \
                  'Loss_hm: {loss_3d.val:.7f} ({loss_3d.avg:.7f})\t' \
                  'Loss_paf: {loss_paf.val:.7f} ({loss_paf.avg:.7f})\t' \
                  'Loss_mask: {loss_cord.val:.6f} ({loss_cord.avg:.6f})\t' \
                  'Loss_fore_depth: {loss_fore_depth.val:.6f}({loss_fore_depth.avg:.6f})\t' \
                  'Memory {memory:.1f}'.format(
                    epoch, i, len(loader), batch_time=batch_time,
                    speed=image.size(0) / batch_time.val,
                    data_time=data_time, loss=losses, loss_2d=losses_depth, loss_3d=losses_hm, loss_fore_depth = losses_fore_depth, loss_paf = losses_paf,
                    loss_cord=losses_mask, memory=gpu_memory_usage)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_loss_mask', losses_mask.val, global_steps)
            writer.add_scalar('train_loss_hm', losses_hm.val, global_steps)
            writer.add_scalar('train_loss_paf', losses_paf.val, global_steps)
            writer.add_scalar('train_loss_depth', losses_depth.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            # debug file
            if i % 1000 == 0:
                # save fig
                pred = nn.functional.interpolate(pred, depth.shape[-2:], mode='bilinear', align_corners=True)
                pred = pred.detach().cpu().numpy()
                pred[pred < config.DATASET.MIN_DEPTH] = config.DATASET.MIN_DEPTH
                pred[pred > config.DATASET.MAX_DEPTH] = config.DATASET.MAX_DEPTH
                pred[np.isinf(pred)] = config.DATASET.MAX_DEPTH
                pred[np.isnan(pred)] = config.DATASET.MIN_DEPTH

                vis_depth = pred[0,0,...]
                temp_mask = mask_prob[0,0,...].detach().cpu().numpy()
                vis_mask = (temp_mask>0.5).astype(np.int)
                # vis_uncertainty = depth_uncertainty[0,0,...]
                # vis_uncertainty = vis_uncertainty.detach().cpu().numpy()
                # vis_hm = heatmap[0,0,...].detach().cpu().numpy()
                folder_name = os.path.join(output_dir, 'debug_train_pics')
                depth_folder = os.path.join(folder_name, 'depth')
                hm_folder = os.path.join(folder_name,'heatmap')
                mask_folder = os.path.join(folder_name,'mask')
                # uncertainty_folder = os.path.join(folder_name,'uncertainty')
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)
                if not os.path.exists(depth_folder):
                    os.makedirs(depth_folder)
                if not os.path.exists(hm_folder):
                    os.makedirs(hm_folder)
                if not os.path.exists(mask_folder):
                    os.makedirs(mask_folder)
                # if not os.path.exists(uncertainty_folder):
                #     os.makedirs(uncertainty_folder)
                vis_pic = tensor2im(image[0,...])
                for hm_idx in range(number_joints):
                    vis_hm = heatmap[0,hm_idx,...].detach().cpu().numpy()
                    vis_hm_gt = hm_gt[0,hm_idx,...].detach().cpu().numpy()
                    # vis_out = np.concatenate([vis_hm, vis_hm_gt], axis = -1)
                    fig = plt.figure()
                    plt.subplot(131)
                    plt.imshow(vis_hm)
                    plt.subplot(132)
                    plt.imshow(vis_hm_gt)
                    plt.subplot(133)
                    plt.imshow(vis_pic)
                    plt.savefig(os.path.join(hm_folder, f'hm_{epoch}_i_{i}_joint_{hm_idx}.jpg'))
                    plt.clf()
                    plt.close(fig)
                
                fig = plt.figure()
                plt.imshow(vis_depth,cmap='magma_r')
                plt.savefig(os.path.join(depth_folder, f'depth_{epoch}_i_{i}.jpg'))
                plt.clf()
                plt.close(fig)
                fig = plt.figure()
                plt.imshow(vis_mask)
                plt.savefig(os.path.join(mask_folder, f'mask_{epoch}_i_{i}.jpg'))
                plt.clf()
                plt.close(fig)
                # fig = plt.figure()
                # plt.imshow(vis_uncertainty,cmap='plasma')    #,cmap='magma_r'
                # plt.savefig(os.path.join(uncertainty_folder, f'uncertainty_{epoch}_i_{i}.jpg'))
                # plt.clf()
                # plt.close(fig)
                





def validate_depth(config, model, loader, output_dir, epoch=0, vali=False, device=torch.device('cuda'), logger=logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_depth = AverageMeter()
    losses_hm = AverageMeter()
    losses_mask = AverageMeter()
    losses_fore_depth = AverageMeter()
    losses_paf = AverageMeter()

    metrics_c = RunningAverageDict()
    criterion_hm = PerJointMSELoss().cuda()
    criterion_dense = SILogLoss().cuda()
    criterion_chamfer = BinsChamferLoss().cuda()
    criterion_mask = CrossEntropyMaskLoss().cuda()
    criterion_fore_depth = foreground_depth_loss().cuda()
    
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, batch in enumerate(loader):
            if len(batch) == 0:
                continue
            data_time.update(time.time() - end)
            image, depth, mask_gt, hm_gt, wt_gt, paf_gt = batch
            image = image.to(device)
            depth = depth.to(device)
            mask_gt = mask_gt.to(device)
            hm_gt = hm_gt.to(device)
            wt_gt = wt_gt.to(device)
            paf_gt = paf_gt.to(device)

            bin_edges, pred, heatmap, paf_pred, mask_prob, feature_out = model(image) #uncertainty ,

            # vis_mask = (mask_prob > 0.5).astype(np.int)

            depth_mask = depth > config.DATASET.MIN_DEPTH
            attention_mask = depth_mask * mask_gt
            l_dense = criterion_dense(pred, depth, mask=depth_mask.to(torch.bool), interpolate=True) #uncertainty,
            losses_depth.update(l_dense.item())
            l_fore_depth = criterion_fore_depth(pred, depth, mask=attention_mask.to(torch.bool), interpolate=True)
            losses_fore_depth.update(l_fore_depth.item())
            pred = nn.functional.interpolate(pred, depth.shape[-2:], mode='bilinear', align_corners=True)

            pred = pred.detach().cpu().numpy()
            pred[pred < config.DATASET.MIN_DEPTH] = config.DATASET.MIN_DEPTH
            pred[pred > config.DATASET.MAX_DEPTH] = config.DATASET.MAX_DEPTH
            pred[np.isinf(pred)] = config.DATASET.MAX_DEPTH
            pred[np.isnan(pred)] = config.DATASET.MIN_DEPTH

            depth_gt = depth.cpu().numpy()
            valid_mask = np.logical_and(depth_gt > config.DATASET.MIN_DEPTH, depth_gt < config.DATASET.MAX_DEPTH)

            metrics_c.update(compute_errors(depth_gt[valid_mask], pred[valid_mask]))


            batch_time.update(time.time() - end)
            end = time.time()
            if i % config.PRINT_FREQ == 0 or i == len(loader) - 1:
                gpu_memory_usage = torch.cuda.memory_allocated(0)
                msg = 'Test: [{0}/{1}]\t' \
                      'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed: {speed:.1f} samples/s\t' \
                      'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'loss_fore: {losses_fore_depth.val:.3f} ({losses_fore_depth.avg:.3f})\t' \
                      'Memory {memory:.1f}'.format(
                        i, len(loader), batch_time=batch_time,
                        speed=image.size(0) / batch_time.val,
                        data_time=data_time,losses_fore_depth = losses_fore_depth ,memory=gpu_memory_usage)
                logger.info(msg)

                if i % 1000 == 0:
                    # save fig
                    vis_depth = pred[0,0,...]
                    temp_mask = mask_prob[0,0,...].detach().cpu().numpy()
                    vis_mask = (temp_mask>0.5).astype(np.int)
                    vis_hm = heatmap[0,0,...].detach().cpu().numpy()
                    # vis_uncertainty = uncertainty[0,0,...].detach().cpu().numpy()
                    folder_name = os.path.join(output_dir, 'debug_test_pics')
                    depth_folder = os.path.join(folder_name, 'depth')
                    hm_folder = os.path.join(folder_name,'heatmap')
                    mask_folder = os.path.join(folder_name,'mask')
                    # uncertainty_folder = os.path.join(folder_name,'uncertainty')
                    if not os.path.exists(folder_name):
                        os.makedirs(folder_name)
                    if not os.path.exists(depth_folder):
                        os.makedirs(depth_folder)
                    if not os.path.exists(hm_folder):
                        os.makedirs(hm_folder)
                    if not os.path.exists(mask_folder):
                        os.makedirs(mask_folder)
                    # if not os.path.exists(uncertainty_folder):
                    #     os.makedirs(uncertainty_folder)    
                    plt.imshow(vis_hm)
                    plt.savefig(os.path.join(hm_folder, f'hm_{epoch}_i_{i}.jpg'))
                    plt.imshow(vis_depth,cmap='magma_r')
                    plt.savefig(os.path.join(depth_folder, f'depth_{epoch}_i_{i}.jpg'))
                    plt.imshow(vis_mask)
                    plt.savefig(os.path.join(mask_folder, f'mask_{epoch}_i_{i}.jpg'))
                    # plt.imshow(vis_uncertainty,cmap='plasma')   #,cmap='magma_r'
                    # plt.savefig(os.path.join(uncertainty_folder, f'uncertainty_{epoch}_i_{i}.jpg'))

    metrics = metrics_c.get_value()
    msg = 'a1: {aps_25:.4f}\ta2: {aps_50:.4f}\ta3: {aps_75:.4f}\t' \
            'abs_rel: {aps_100:.4f}\trmse: {aps_125:.4f}\tlog10: {aps_150:.4f}\t' \
            'rmse_log: {recall:.4f}\tsq_rel: {mpjpe:.3f}'.format(
            aps_25=metrics['a1'], aps_50=metrics['a2'], aps_75=metrics['a3'], aps_100=metrics['abs_rel'],
            aps_125=metrics['rmse'], aps_150=metrics['log_10'], recall=metrics['rmse_log'], mpjpe=metrics['sq_rel']
            )
    logger.info(msg)
    return metrics

def train_points3d_bak(config, model, optimizer, loader, epoch, output_dir, writer_dict, device=torch.device('cuda'), dtype=torch.float):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_2d = AverageMeter()
    losses_depth = AverageMeter()
    losses_hm = AverageMeter()
    losses_mask = AverageMeter()
    losses_fore_depth = AverageMeter()

    metrics_c = RunningAverageDict()
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
    points_extractor = get_3d_points(orig_W=orig_w, orig_H = orig_h).cuda()

    
    model.train()
    end = time.time()
    for i, batch in enumerate(loader):
        if len(batch) == 0:
            continue
        data_time.update(time.time() - end)
        output_img, output_depth, output_valid_mask,output_hm, output_weights, output_trans, output_K, output_M, output_diff, output_3d_pose = batch
        batch_num = output_img.shape[0]
        view_num = output_img.shape[1]
        number_joints = config.NETWORK.NUM_JOINTS
        # process in multiple view form
        # preset the loss
        loss = torch.tensor(0).to(device)
        total_points = [[[] for _ in range(batch_num)] for _ in range(number_joints)]
        for view in range(view_num): 
            image = output_img[:,view,...]
            depth = output_depth[:,view,...]
            mask_gt = output_valid_mask[:,view,...]
            hm_gt = output_hm[:,view,...]
            wt_gt = output_weights[:,view,...]
            trans_matrix = output_trans[:,view,...]
            # trans_matrix = trans_matrix.numpy()
            K_matrix = output_K[:,view,...]
            # K_matrix = K_matrix.numpy()
            M_matrix = output_M[:,view,...]
            # M_matrix = M_matrix.numpy()
            diff = output_diff[:,view,...]
            

            image = image.to(device)
            depth = depth.to(device)
            mask_gt = mask_gt.to(device)
            hm_gt = hm_gt.to(device)
            wt_gt = wt_gt.to(device)
            trans_matrix = trans_matrix.to(device)
            K_matrix = K_matrix.to(device)
            M_matrix = M_matrix.to(device)
            diff = diff.to(device)
            


            bin_edges, pred, heatmap, mask_prob, feature_out = model(image)

            # depth loss
            depth_mask = depth > config.DATASET.MIN_DEPTH
            attention_mask = depth_mask * mask_gt   
            l_dense = criterion_dense(pred, depth, mask=depth_mask.to(torch.bool), interpolate=True)
            l_dense_attention = criterion_dense_attention(pred, depth, mask=attention_mask.to(torch.bool), interpolate=True)
            l_chamfer = criterion_chamfer(bin_edges, depth.float())

            l_fore_depth =criterion_fore_depth(pred, depth, mask=attention_mask.to(torch.bool), interpolate=True)
            losses_fore_depth.update(l_fore_depth.item())
            loss_depth = l_dense + 0.1 * l_chamfer +  l_dense_attention # pay more attention to the  #+ 2e-3 * l_grad # pay attention to the foreground and control the grad
            losses_depth.update(loss_depth.item())
            
            # hm loss
            loss_hm = criterion_hm(heatmap, hm_gt, mask_gt ,True, wt_gt)
            losses_hm.update(loss_hm.item() * 50)
            
             # mask loss
            loss_mask = criterion_mask(mask_prob, mask_gt.long())
            losses_mask.update(loss_mask.item() * 5)

            loss_2d = loss_depth + 50 * loss_hm + loss_mask * 5   
            losses_2d.update(loss_2d.item())

            # 2d network can be trained by the loss loss_2d
            # the loss can be added to the backward update
            pred_process = pred.clone() # for projection


            # pred = pred.detach().cpu().numpy()
            # pred[pred < config.DATASET.MIN_DEPTH] = config.DATASET.MIN_DEPTH
            # pred[pred > config.DATASET.MAX_DEPTH] = config.DATASET.MAX_DEPTH
            # pred[np.isinf(pred)] = config.DATASET.MAX_DEPTH
            # pred[np.isnan(pred)] = config.DATASET.MIN_DEPTH

            # depth_gt = depth.cpu().numpy()
            # valid_mask = np.logical_and(depth_gt > config.DATASET.MIN_DEPTH, depth_gt < config.DATASET.MAX_DEPTH)

            # 1. get the mask to do the filtering as "number"
            fitted_mask = F.interpolate(mask_prob, scale_factor=0.5)
            # filter mask
            filter_gradient_mask = generator_gradient(pred_process.clone())

            # convolution is 2*P = F_size -1 for same 
            kernel = torch.ones(5,5).to(device)
            mask = (fitted_mask > 0.5).int()
            # erosion mask
            erode_mask = erode(mask,kernel.detach()) # erode error

            filter_mask = []

            for hm_idx in range(number_joints):
                hm_sampling = heatmap[:,hm_idx:hm_idx + 1,:,:]
                hm_max = torch.max(hm_sampling)
                choice_generator = 2 * hm_max * torch.rand(hm_sampling.shape).to(device)
                hm_sampling_mask = hm_sampling > choice_generator
                filter_mask.append(erode_mask * filter_gradient_mask * hm_sampling_mask)

        
            total_points = points_extractor(pred_process, filter_mask, K_matrix, M_matrix,diff, trans_matrix, total_points, feature_out.clone()) # 


            
            
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
                fill_indx = torch.randint(s_num, (offset, ))
                fill_tensor = s_batch_points[fill_indx,:] # 索引可能不够了
                new_tensor = torch.cat([s_batch_points,fill_tensor],dim=0)
                total_points[hm_idx][b] = new_tensor.unsqueeze(0)

            total_points[hm_idx] = torch.cat(total_points[hm_idx],dim=0)

        # send the result into the votenet

        import pdb; pdb.set_trace()


                    
    # return None

def train_points3d(config, model, optimizer, loader, epoch, output_dir, writer_dict, device=torch.device('cuda'), dtype=torch.float, logger=logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_2d = AverageMeter()
    losses_3d = AverageMeter()
    losses_vote = AverageMeter()
    losses_objective = AverageMeter()
    losses_distance = AverageMeter()
    losses_depth = AverageMeter()
    losses_hm = AverageMeter()
    losses_paf = AverageMeter()
    losses_mask = AverageMeter()
    model.train()

    end = time.time()

    for i, batch in enumerate(loader):
        if batch == None:
            continue
        if len(batch) == 0:
            continue
        data_time.update(time.time() - end)
        output_img, output_depth, output_valid_mask,output_hm, output_paf ,output_weights, output_trans, output_K, output_M, output_diff, output_3d_pose,output_num_people = batch
        batch_num,_,num_joints,_ = output_3d_pose.shape
       
        loss_vote,loss_objective,loss_distance, loss_3d, loss_depth, loss_hm, loss_paf,loss_mask ,loss_2d, predicted_3dpose, pred, heatmap, mask_prob, vote_xyz= model(output_img, output_depth, output_valid_mask, output_hm, output_paf,output_weights, output_trans, 
                                            output_K, output_M, output_diff, output_3d_pose, epoch, i)
        
        # if torch.sum(valid) < batch_num:
        #     valid = False
        # else:
        #     valid = True
        if torch.any(torch.isnan(loss_2d)): 
            continue
        
        loss_vote = loss_vote.mean()
        loss_objective = loss_objective.mean()
        loss_distance = loss_distance.mean()
        loss_2d = loss_2d.mean()
        loss_3d = loss_3d.mean()
        loss_paf = loss_paf.mean()
        # loss_3d = loss_3d.mean()
        loss_depth = loss_depth.mean()
        loss_hm = loss_hm.mean()
        loss_mask = loss_mask.mean()

        losses_depth.update(loss_depth.item())
        losses_hm.update(loss_hm.item())
        losses_mask.update(loss_mask.item())
        losses_2d.update(loss_2d.item())
        losses_vote.update(loss_vote.item())
        losses_paf.update(loss_paf.item())
        losses_objective.update(loss_objective.item())
        losses_distance.update(loss_distance.item())
        losses_3d.update(loss_3d.item())

        # if (epoch == 0) and (i <= 3000) and ~valid: # first epoch only 2d valid is for 3d network
        #     import pdb;pdb.set_trace()
        #     loss_total = loss_2d
        #     losses_depth.update(loss_depth.item())
        #     losses_hm.update(loss_hm.item())
        #     losses_mask.update(loss_mask.item())
        #     losses_2d.update(loss_2d.item())
        # else:
        #     if loss_vote == 0:
        #         loss_total = loss_2d
        #         losses_depth.update(loss_depth.item())
        #         losses_hm.update(loss_hm.item())
        #         losses_mask.update(loss_mask.item())
        #         losses_2d.update(loss_2d.item())
        #     else:
        #         if loss_distance == 0:
        #             loss_3d = loss_vote + loss_objective
        #             loss_total = loss_2d + loss_vote + loss_objective
        #             losses_vote.update(loss_vote.item())
        #             losses_objective.update(loss_objective.item())
        #             losses_depth.update(loss_depth.item())
        #             losses_hm.update(loss_hm.item())
        #             losses_mask.update(loss_mask.item())
        #             losses_2d.update(loss_2d.item())
        #         else:
        #             loss_3d = loss_vote + loss_objective + loss_distance 
        #             loss_total = loss_2d + loss_3d
        #             losses_depth.update(loss_depth.item())
        #             losses_hm.update(loss_hm.item())
        #             losses_mask.update(loss_mask.item())
        #             losses_2d.update(loss_2d.item())
        #             losses_vote.update(loss_vote.item())
        #             losses_objective.update(loss_objective.item())
        #             losses_distance.update(loss_distance.item())
        #             losses_3d.update(loss_3d.item())
        
        
        

        optimizer.zero_grad()
        (loss_3d).backward() # add loss 3d #TODO: loss 3d for debug + loss_2d
        optimizer.step()

        data_time.update(time.time() - end)
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            gpu_memory_usage = torch.cuda.memory_allocated(0)
            msg =   'Epoch: [{0}][{1}/{2}]\t' \
                    'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                    'Speed: {speed:.1f} samples/s\t' \
                    'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                    'loss_vote {losses_vote.val:.3f} ({losses_vote.avg:.3f})\t' \
                    'loss_objective {losses_objective.val:.3f} ({losses_objective.avg:.3f})\t' \
                    'loss_distance {losses_distance.val:.3f} ({losses_distance.avg:.3f})\t' \
                    'loss_3d: {losses_3d.val:.3f} ({losses_3d.avg:.3f})\t' \
                    'loss_2d: {losses_2d.val:.3f} ({losses_2d.avg:.3f})\t' \
                    'loss_depth: {losses_depth.val:.3f} ({losses_depth.avg:.3f})\t' \
                    'loss_hm: {losses_hm.val:.3f} ({losses_hm.avg:.3f})\t' \
                    'loss_mask: {losses_mask.val:.3f} ({losses_mask.avg:.3f})\t' \
                    'Memory {memory:.1f}'.format(
                    epoch, i, len(loader), batch_time=batch_time,
                    speed=output_img.size(0) / batch_time.val,
                    data_time=data_time, losses_2d = losses_2d, 
                    losses_3d = losses_3d,losses_vote= losses_vote, 
                    losses_objective = losses_objective,losses_distance = losses_distance,
                    losses_depth =losses_depth, losses_hm = losses_hm,losses_mask = losses_mask,
                    memory=gpu_memory_usage)
            
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_2d', losses_2d.val, global_steps)
            writer.add_scalar('train_3d', losses_3d.val, global_steps)
            writer.add_scalar('train_vote', losses_vote.val, global_steps)
            writer.add_scalar('train_object', losses_objective.val, global_steps)
            writer.add_scalar('train_dist', losses_distance.val, global_steps)
            writer.add_scalar('train_loss_hm', losses_hm.val, global_steps)
            writer.add_scalar('train_loss_depth', losses_depth.val, global_steps)
            writer.add_scalar('train_loss_mask', losses_mask.val, global_steps)
            writer.add_scalar('train_loss_paf', losses_paf.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1
            # import pdb; pdb.set_trace()
            if i % 500 == 0:
                # save fig
                
                pred = pred.clone().detach().cpu().numpy()
                pred[pred < config.DATASET.MIN_DEPTH] = config.DATASET.MIN_DEPTH
                pred[pred > config.DATASET.MAX_DEPTH] = config.DATASET.MAX_DEPTH
                pred[np.isinf(pred)] = config.DATASET.MAX_DEPTH
                pred[np.isnan(pred)] = config.DATASET.MIN_DEPTH
                vis_depth = pred[0,0,...]
                temp_mask = mask_prob[0,0,...].clone().detach().cpu().numpy()
                vis_mask = (temp_mask>0.5).astype(np.int)
                poshm = torch.randint(config.NETWORK.NUM_JOINTS,(1,))
                vis_hm = heatmap[0,poshm.item(),...].clone().detach().cpu().numpy() 
                folder_name = os.path.join(output_dir, 'debug_train_pics')
                depth_folder = os.path.join(folder_name, 'depth')
                hm_folder = os.path.join(folder_name,'heatmap')
                mask_folder = os.path.join(folder_name,'mask')
                vote_xyz_folder = os.path.join(folder_name,'joints')
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)
                if not os.path.exists(depth_folder):
                    os.makedirs(depth_folder)
                if not os.path.exists(hm_folder):
                    os.makedirs(hm_folder)
                if not os.path.exists(mask_folder):
                    os.makedirs(mask_folder)
                if not os.path.exists(vote_xyz_folder):
                    os.makedirs(vote_xyz_folder) 
                fig = plt.figure()
                ax2 = plt.axes(projection='3d')
                
                vote_xyz_vis = vote_xyz[0].detach().cpu().numpy()
                poshm = torch.randint(config.NETWORK.NUM_JOINTS,(1,))
                ax2.scatter3D(vote_xyz_vis[:,poshm.item(),0],vote_xyz_vis[:,poshm.item(),2],-vote_xyz_vis[:,poshm.item(),1], s=5, color=[1,0,0]) # plot the first keyp
                pose_gt = output_3d_pose[:,:,poshm.item(),:3].float().to(device) / 100
                gt_center = pose_gt[0]
                xd_gt = gt_center[:,0].detach().cpu().numpy()
                yd_gt = gt_center[:,1].detach().cpu().numpy()
                zd_gt = gt_center[:,2].detach().cpu().numpy()
                ax2.scatter3D(xd_gt,zd_gt,-yd_gt,s=20, color=[0,0,1])
                plt.savefig(os.path.join(vote_xyz_folder, f'points_{epoch}_i_{i}_pose_{poshm.item()}.jpg'))
                plt.clf()
                plt.close(fig) 
                
                fig = plt.figure()
                plt.imshow(vis_hm)
                plt.savefig(os.path.join(hm_folder, f'hm_{epoch}_i_{i}_hm_{poshm.item()}.jpg'))
                plt.clf()
                plt.close(fig)
                fig = plt.figure()
                plt.imshow(vis_depth,cmap='magma_r')
                plt.savefig(os.path.join(depth_folder, f'depth_{epoch}_i_{i}.jpg'))
                plt.clf()
                plt.close(fig)
                fig = plt.figure()
                plt.imshow(vis_mask)
                plt.savefig(os.path.join(mask_folder, f'mask_{epoch}_i_{i}.jpg'))
                plt.clf()
                plt.close(fig)
                hm_valid_num = len(predicted_3dpose)
                # hm_valid_num = 19
                fig = plt.figure()
                ax1 = plt.axes(projection='3d')
                # if (epoch or (i > 3000)) and valid: # epoch 0 did not consider the 3d network
                for hm in range(num_joints):
                    center_point = predicted_3dpose[:,hm,:,:]

                    # predicted_3dpose[hm]
                    # center_point = extracted['center']

                    pose_gt = output_3d_pose[:,:,hm,:3].float().to(device) / 100
                    _,_,_,ind2 = nn_distance(center_point,pose_gt)
                    # by default we only plot the first batch
                    predicted_center = center_point[0]
                    gt_center = pose_gt[0]
                    xd_pred = predicted_center[ind2[0],0].detach().cpu().numpy()
                    yd_pred = predicted_center[ind2[0],1].detach().cpu().numpy()
                    zd_pred = predicted_center[ind2[0],2].detach().cpu().numpy()
                    ax1.scatter3D(xd_pred,zd_pred,-yd_pred, s=5, color=[1,0,0])
                    xd_gt = gt_center[:,0].detach().cpu().numpy()
                    yd_gt = gt_center[:,1].detach().cpu().numpy()
                    zd_gt = gt_center[:,2].detach().cpu().numpy()
                    ax1.scatter3D(xd_gt,zd_gt,-yd_gt,s=7, color=[0,0,1])
                    # plot the predicted in red and gt in blue
                # folder_name = os.path.join(output_dir, 'debug_train_pics')
                for kp1,kp2 in CONNS:
                    center_point_1 = predicted_3dpose[:,kp1,:,:]
                    center_point_2 = predicted_3dpose[:,kp2,:,:]
                    pose_gt_1 = output_3d_pose[:,:,kp1,:3].float().to(device) / 100
                    pose_gt_2 = output_3d_pose[:,:,kp2,:3].float().to(device) / 100
                    _,_,_,ind1 = nn_distance(center_point_1,pose_gt_1)
                    _,_,_,ind2 = nn_distance(center_point_2,pose_gt_2)
                    predicted_center_1 = center_point_1[0]
                    predicted_center_2 = center_point_2[0]
                    gt_center_1 = pose_gt_1[0]
                    gt_center_2 = pose_gt_2[0]
                    # pred_result
                    xd_pred_1 = predicted_center_1[ind1[0],0].detach().cpu().numpy()
                    yd_pred_1 = predicted_center_1[ind1[0],1].detach().cpu().numpy()
                    zd_pred_1 = predicted_center_1[ind1[0],2].detach().cpu().numpy()
                    xd_pred_2 = predicted_center_2[ind2[0],0].detach().cpu().numpy()
                    yd_pred_2 = predicted_center_2[ind2[0],1].detach().cpu().numpy()
                    zd_pred_2 = predicted_center_2[ind2[0],2].detach().cpu().numpy()
                    # gt_result
                    xd_gt_1 = gt_center_1[:,0].detach().cpu().numpy()
                    yd_gt_1 = gt_center_1[:,1].detach().cpu().numpy()
                    zd_gt_1 = gt_center_1[:,2].detach().cpu().numpy()
                    xd_gt_2 = gt_center_2[:,0].detach().cpu().numpy()
                    yd_gt_2 = gt_center_2[:,1].detach().cpu().numpy()
                    zd_gt_2 = gt_center_2[:,2].detach().cpu().numpy()
                    # plotting each joints with number_of people
                    num_people = len(ind1[0])
                    for num_idx in range(num_people):
                        ax1.plot3D([xd_pred_1[num_idx],xd_pred_2[num_idx]],[zd_pred_1[num_idx],zd_pred_2[num_idx]],[-yd_pred_1[num_idx],-yd_pred_2[num_idx]],color=[1,0,0])
                        ax1.plot3D([xd_gt_1[num_idx],xd_gt_2[num_idx]],[zd_gt_1[num_idx],zd_gt_2[num_idx]],[-yd_gt_1[num_idx],-yd_gt_2[num_idx]],color=[0,0,1])
                
                points_folder = os.path.join(folder_name, 'points')
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)
                if not os.path.exists(points_folder):
                    os.makedirs(points_folder)
                plt.savefig(os.path.join(points_folder, f'points_{epoch}_i_{i}.jpg'))
                plt.clf()
                plt.close(fig)



def validate_points3d(config, model, loader, output_dir, epoch=0, vali=False, device=torch.device('cuda'), dtype=torch.float, logger=logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_2d = AverageMeter()
    losses_3d = AverageMeter()
    losses_vote = AverageMeter()
    losses_objective = AverageMeter()
    losses_distance = AverageMeter()
    losses_depth = AverageMeter()
    losses_hm = AverageMeter()
    losses_paf = AverageMeter()
    losses_mask = AverageMeter()
    losses_total = AverageMeter()
    mpjpe = AverageMeter()
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, batch in enumerate(loader):
            if batch == None:
                continue
            if len(batch) == 0:
                continue
            data_time.update(time.time() - end)
            output_img, output_depth, output_valid_mask,output_hm, output_paf,output_weights, output_trans, output_K, output_M, output_diff, output_3d_pose, output_num_people = batch
            batch_num,_,num_joints,_ = output_3d_pose.shape
            loss_vote,loss_objective,loss_distance, loss_3d, loss_depth, loss_hm, loss_paf,loss_mask ,loss_2d, predicted_3dpose, pred, heatmap, mask_prob, vote_xyz= model(output_img, output_depth, output_valid_mask, output_hm, output_paf,output_weights, output_trans, 
                                            output_K, output_M, output_diff, output_3d_pose,epoch, i)
            
            # if torch.sum(valid) < batch_num:
            #     valid = False
            # else:
            #     valid = True

            if torch.any(torch.isnan(loss_2d)): 
                continue
            loss_vote = loss_vote.mean()
            loss_objective = loss_objective.mean()
            loss_distance = loss_distance.mean()
            loss_2d = loss_2d.mean()
            loss_3d = loss_3d.mean()
            loss_paf = loss_paf.mean() 
            loss_depth = loss_depth.mean()
            loss_hm = loss_hm.mean()
            loss_mask = loss_mask.mean()

            losses_depth.update(loss_depth.item())
            losses_hm.update(loss_hm.item())
            losses_paf.update(loss_paf.item())
            losses_mask.update(loss_mask.item())
            losses_2d.update(loss_2d.item())
            losses_vote.update(loss_vote.item())
            losses_objective.update(loss_objective.item())
            losses_distance.update(loss_distance.item())
            losses_3d.update(loss_3d.item())

            # if epoch == 0: # first epoch only 2d
            #     loss_total = loss_2d
            #     losses_depth.update(loss_depth.item())
            #     losses_hm.update(loss_hm.item())
            #     losses_mask.update(loss_mask.item())
            #     losses_2d.update(loss_2d.item())
            # else:

            # if ~ valid: # no grad 的原因
            #     loss_total = loss_2d
            #     losses_depth.update(loss_depth.item())
            #     losses_hm.update(loss_hm.item())
            #     losses_mask.update(loss_mask.item())
            #     losses_2d.update(loss_2d.item())
            # else:
            #     if loss_vote == 0:
            #         loss_total = loss_2d
            #         losses_depth.update(loss_depth.item())
            #         losses_hm.update(loss_hm.item())
            #         losses_mask.update(loss_mask.item())
            #         losses_2d.update(loss_2d.item())
            #     else:
            #         if losses_distance == 0:
            #             # loss_total = loss_2d + loss_vote + loss_objective
            #             losses_vote.update(loss_vote.item())
            #             losses_objective.update(loss_objective.item())
            #             losses_depth.update(loss_depth.item())
            #             losses_hm.update(loss_hm.item())
            #             losses_mask.update(loss_mask.item())
            #             losses_2d.update(loss_2d.item())
            #         else:
            #             loss_3d = loss_vote + loss_objective + loss_distance 
            #             # loss_total = loss_2d + loss_3d
            #             losses_depth.update(loss_depth.item())
            #             losses_hm.update(loss_hm.item())
            #             losses_mask.update(loss_mask.item())
            #             losses_2d.update(loss_2d.item())
            #             losses_vote.update(loss_vote.item())
            #             losses_objective.update(loss_objective.item())
            #             losses_distance.update(loss_distance.item())
            #             losses_3d.update(loss_3d.item())
            
            # calculate the mpjpe metric in torch、
            mpjpes = []
            for hm in range(num_joints):
                center_point = predicted_3dpose[:,hm,:,:]
                output_vis = output_3d_pose[:,:,hm,3].float().to(device)
                output_vis_select = (output_vis > 0.1)
                pose_gt = output_3d_pose[:,:,hm,:3].float().to(device) / 100
                _,_,_,ind2 = nn_distance(center_point,pose_gt) # B * M
                # process by batch
                batch_mpjpe = []
                for b_idx in range(batch_num):
                    batch_prop = center_point[b_idx]
                    gt_center = pose_gt[b_idx]
                    gt_vis = output_vis_select[b_idx]
                    gt_people_num = output_num_people[b_idx]
                    gt_vis = gt_vis[:gt_people_num]
                    if torch.sum(gt_vis) == 0:
                        continue
                    indx = ind2[b_idx]
                    pred_pro = batch_prop[indx,:]
                    # calculate the diff
                    diff = pred_pro - gt_center
                    diff = diff[:gt_people_num]
                    diff = diff[gt_vis]
                    mpjpe_b =torch.mean(torch.sqrt(torch.sum(diff ** 2, dim=-1)))
                    batch_mpjpe.append(mpjpe_b.unsqueeze(0))
                if len(batch_mpjpe) == 0:
                    continue
                batch_mpjpe = torch.cat(batch_mpjpe, dim=0)
                hm_mpjpe = torch.mean(batch_mpjpe)
                mpjpes.append(hm_mpjpe.unsqueeze(0))

            if len(mpjpes) > 0:  
                mpjpes = torch.cat(mpjpes, dim=0)
                report_mpjpe = torch.mean(mpjpes)
                mpjpe.update(report_mpjpe.item())

            data_time.update(time.time() - end)
            batch_time.update(time.time() - end)
            end = time.time()

            if i == len(loader) - 1:
                msg = f'---------------mpjpe final: {mpjpe.avg:.3f} ----------------'
                logger.info(msg)


            if i % config.PRINT_FREQ == 0:
                gpu_memory_usage = torch.cuda.memory_allocated(0)
                msg =   'TEST: [{0}][{1}/{2}]\t' \
                        'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                        'Speed: {speed:.1f} samples/s\t' \
                        'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                        'loss_vote {losses_vote.val:.3f} ({losses_vote.avg:.3f})\t' \
                        'loss_objective {losses_objective.val:.3f} ({losses_objective.avg:.3f})\t' \
                        'loss_distance {losses_distance.val:.3f} ({losses_distance.avg:.3f})\t' \
                        'loss_3d: {losses_3d.val:.3f} ({losses_3d.avg:.3f})\t' \
                        'loss_2d: {losses_2d.val:.3f} ({losses_2d.avg:.3f})\t' \
                        'loss_depth: {losses_depth.val:.3f} ({losses_depth.avg:.3f})\t' \
                        'loss_hm: {losses_hm.val:.3f} ({losses_hm.avg:.3f})\t' \
                        'loss_mask: {losses_mask.val:.3f} ({losses_mask.avg:.3f})\t' \
                        'mpjpe: {mpjpe.val:.3f} ({mpjpe.avg:.3f})\t' \
                        'Memory {memory:.1f}'.format(
                        epoch, i, len(loader), batch_time=batch_time,
                        speed=output_img.size(0) / batch_time.val,
                        data_time=data_time, losses_2d = losses_2d, 
                        losses_3d = losses_3d,losses_vote= losses_vote, 
                        losses_objective = losses_objective,losses_distance = losses_distance,
                        losses_depth =losses_depth, losses_hm = losses_hm,losses_mask = losses_mask, mpjpe = mpjpe,
                        memory=gpu_memory_usage)
                
                logger.info(msg)
                # add the vis part to the test
                if i % 300 == 0:
                    pred = pred.clone().detach().cpu().numpy()
                    pred[pred < config.DATASET.MIN_DEPTH] = config.DATASET.MIN_DEPTH
                    pred[pred > config.DATASET.MAX_DEPTH] = config.DATASET.MAX_DEPTH
                    pred[np.isinf(pred)] = config.DATASET.MAX_DEPTH
                    pred[np.isnan(pred)] = config.DATASET.MIN_DEPTH
                    vis_depth = pred[0,0,...]
                    temp_mask = mask_prob[0,0,...].clone().detach().cpu().numpy()
                    vis_mask = (temp_mask>0.5).astype(np.int)
                    vis_hm = heatmap[0,0,...].clone().detach().cpu().numpy()
                    folder_name = os.path.join(output_dir, 'debug_test_pics')
                    depth_folder = os.path.join(folder_name, 'depth')
                    hm_folder = os.path.join(folder_name,'heatmap')
                    mask_folder = os.path.join(folder_name,'mask')
                    vote_xyz_folder = os.path.join(folder_name,'joints')
                    if not os.path.exists(folder_name):
                        os.makedirs(folder_name)
                    if not os.path.exists(depth_folder):
                        os.makedirs(depth_folder)
                    if not os.path.exists(hm_folder):
                        os.makedirs(hm_folder)
                    if not os.path.exists(mask_folder):
                        os.makedirs(mask_folder)
                    if not os.path.exists(vote_xyz_folder):
                        os.makedirs(vote_xyz_folder)    
                    fig = plt.figure()
                    ax2 = plt.axes(projection='3d')
                    # import pdb; pdb.set_trace()
                    vote_xyz_vis = vote_xyz[0].detach().cpu().numpy()
                    poshm = torch.randint(config.NETWORK.NUM_JOINTS,(1,))
                    ax2.scatter3D(vote_xyz_vis[:,poshm.item(),0],vote_xyz_vis[:,poshm.item(),2],-vote_xyz_vis[:,poshm.item(),1], s=5, color=[1,0,0]) # plot the first keyp
                    pose_gt = output_3d_pose[:,:,poshm.item(),:3].float().to(device) / 100
                    gt_center = pose_gt[0]
                    xd_gt = gt_center[:,0].detach().cpu().numpy()
                    yd_gt = gt_center[:,1].detach().cpu().numpy()
                    zd_gt = gt_center[:,2].detach().cpu().numpy()
                    ax2.scatter3D(xd_gt,zd_gt,-yd_gt,s=20, color=[0,0,1])
                    plt.savefig(os.path.join(vote_xyz_folder, f'points_{epoch}_i_{i}_pose_{poshm.item()}.jpg'))
                    plt.clf()
                    plt.close(fig) 

                    fig = plt.figure()
                    plt.imshow(vis_hm)
                    plt.savefig(os.path.join(hm_folder, f'hm_{epoch}_i_{i}.jpg'))
                    plt.clf()
                    plt.close(fig) 
                    fig = plt.figure()
                    plt.imshow(vis_depth,cmap='magma_r')
                    plt.savefig(os.path.join(depth_folder, f'depth_{epoch}_i_{i}.jpg'))
                    plt.clf()
                    plt.close(fig) 
                    fig = plt.figure()
                    plt.imshow(vis_mask)
                    plt.savefig(os.path.join(mask_folder, f'mask_{epoch}_i_{i}.jpg'))
                    plt.clf()
                    plt.close(fig)
                    # hm_valid_num = len(predicted_3dpose)
                    fig = plt.figure()
                    ax1 = plt.axes(projection='3d')
                    # if valid:
                    for hm in range(num_joints):
                        center_point = predicted_3dpose[:,hm,:,:]
                        # if torch.sum(center_point) == 0:
                        #     continue
                        # import pdb;pdb.set_trace()
                        # center_point = end_point_hm # B,M,3
                        # 跟output 对齐
                        pose_gt = output_3d_pose[:,:,hm,:3].float().to(device) / 100
                        _,_,_,ind2 = nn_distance(center_point,pose_gt)
                        # by default we only plot the first batch
                        predicted_center = center_point[0]
                        gt_center = pose_gt[0]
                        xd_pred = predicted_center[ind2[0],0].detach().cpu().numpy()
                        yd_pred = predicted_center[ind2[0],1].detach().cpu().numpy()
                        zd_pred = predicted_center[ind2[0],2].detach().cpu().numpy()
                        ax1.scatter3D(xd_pred,zd_pred,-yd_pred, s=5, color=[1,0,0])

                        xd_gt = gt_center[:,0].detach().cpu().numpy()
                        yd_gt = gt_center[:,1].detach().cpu().numpy()
                        zd_gt = gt_center[:,2].detach().cpu().numpy()
                        ax1.scatter3D(xd_gt,zd_gt,-yd_gt,s=7, color=[0,0,1])
                        # plot the predicted in red and gt in blue
                    
                    for kp1,kp2 in CONNS:
                        center_point_1 = predicted_3dpose[:,kp1,:,:]
                        center_point_2 = predicted_3dpose[:,kp2,:,:]
                        pose_gt_1 = output_3d_pose[:,:,kp1,:3].float().to(device) / 100
                        pose_gt_2 = output_3d_pose[:,:,kp2,:3].float().to(device) / 100
                        _,_,_,ind1 = nn_distance(center_point_1,pose_gt_1)
                        _,_,_,ind2 = nn_distance(center_point_2,pose_gt_2)
                        predicted_center_1 = center_point_1[0]
                        predicted_center_2 = center_point_2[0]
                        gt_center_1 = pose_gt_1[0]
                        gt_center_2 = pose_gt_2[0]
                        # pred_result
                        xd_pred_1 = predicted_center_1[ind1[0],0].detach().cpu().numpy()
                        yd_pred_1 = predicted_center_1[ind1[0],1].detach().cpu().numpy()
                        zd_pred_1 = predicted_center_1[ind1[0],2].detach().cpu().numpy()
                        xd_pred_2 = predicted_center_2[ind2[0],0].detach().cpu().numpy()
                        yd_pred_2 = predicted_center_2[ind2[0],1].detach().cpu().numpy()
                        zd_pred_2 = predicted_center_2[ind2[0],2].detach().cpu().numpy()
                        # gt_result
                        xd_gt_1 = gt_center_1[:,0].detach().cpu().numpy()
                        yd_gt_1 = gt_center_1[:,1].detach().cpu().numpy()
                        zd_gt_1 = gt_center_1[:,2].detach().cpu().numpy()
                        xd_gt_2 = gt_center_2[:,0].detach().cpu().numpy()
                        yd_gt_2 = gt_center_2[:,1].detach().cpu().numpy()
                        zd_gt_2 = gt_center_2[:,2].detach().cpu().numpy()
                        # plotting each joints with number_of people
                        num_people = len(ind1[0])
                        for num_idx in range(num_people):
                            ax1.plot3D([xd_pred_1[num_idx],xd_pred_2[num_idx]],[zd_pred_1[num_idx],zd_pred_2[num_idx]],[-yd_pred_1[num_idx],-yd_pred_2[num_idx]],color=[1,0,0])
                            ax1.plot3D([xd_gt_1[num_idx],xd_gt_2[num_idx]],[zd_gt_1[num_idx],zd_gt_2[num_idx]],[-yd_gt_1[num_idx],-yd_gt_2[num_idx]],color=[0,0,1])

                    folder_name = os.path.join(output_dir, 'debug_test_pics')
                    points_folder = os.path.join(folder_name, 'points')
                    if not os.path.exists(folder_name):
                        os.makedirs(folder_name)
                    if not os.path.exists(points_folder):
                        os.makedirs(points_folder)
                    plt.savefig(os.path.join(points_folder, f'points_{epoch}_i_{i}.jpg'))
                    plt.clf()
                    plt.close(fig) 
    
    return mpjpe.avg # 先用深度为标准
                








def validate_depth_vis(config, model, loader, output_dir, epoch=0, vali=False, device=torch.device('cuda')): # input the multiview data
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_depth = AverageMeter()
    losses_hm = AverageMeter()
    losses_mask = AverageMeter()
    losses_fore_depth = AverageMeter()

    metrics_c = RunningAverageDict()
    criterion_hm = PerJointMSELoss().cuda()
    criterion_dense = SILogLoss().cuda()
    criterion_chamfer = BinsChamferLoss().cuda()
    criterion_mask = CrossEntropyMaskLoss().cuda()
    criterion_fore_depth = foreground_depth_loss().cuda()
    generator_gradient = grad_generation().cuda()
    erode = erode_generation().cuda()
    orig_w = 1920
    orig_h = 1080
    points_extractor = get_3d_points(orig_W=orig_w, orig_H =orig_h).cuda()

    
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, batch in enumerate(loader):
            if len(batch) == 0:
                continue
            data_time.update(time.time() - end)
            # output_img, output_depth, output_valid_mask,output_hm, output_weights, output_trans, output_K, output_M, output_diff, output_3d_pose = batch
            output_img, output_depth, output_valid_mask,output_hm, output_paf,output_weights, output_trans, output_K, output_M, output_diff, output_3d_pose, output_num_people = batch
            batch_num = output_img.shape[0]
            view_num = output_img.shape[1]
            number_joints = config.NETWORK.NUM_JOINTS
            # process in multiple view form
            total_points = [[] for _ in range(batch_num)]
            gt_total_points = [[] for _ in range(batch_num)]
            for view in range(view_num): 
                image = output_img[:,view,...]
                depth = output_depth[:,view,...]
                mask_gt = output_valid_mask[:,view,...]
                hm_gt = output_hm[:,view,...]
                wt_gt = output_weights[:,view,...]
                trans_matrix = output_trans[:,view,...]
                # trans_matrix = trans_matrix.numpy()
                K_matrix = output_K[:,view,...]
                # K_matrix = K_matrix.numpy()
                M_matrix = output_M[:,view,...]
                # M_matrix = M_matrix.numpy()
                diff = output_diff[:,view,...]
                # diff = diff.numpy()

                image = image.to(device)
                depth = depth.to(device)
                mask_gt = mask_gt.to(device)
                hm_gt = hm_gt.to(device)
                wt_gt = wt_gt.to(device)
                trans_matrix = trans_matrix.to(device)
                K_matrix = K_matrix.to(device)
                M_matrix = M_matrix.to(device)
                diff = diff.to(device)
                

                bin_edges, pred, heatmap, paf_pred, mask_prob, feature_out = model(image)

                # vis_mask = (mask_prob > 0.5).astype(np.int)
                # import pdb; pdb.set_trace()
                depth_mask = depth > config.DATASET.MIN_DEPTH
                attention_mask = depth_mask * mask_gt
                
                
                # l_dense = criterion_dense(pred, uncertainty,depth, mask=depth_mask.to(torch.bool), interpolate=True)
                # losses_depth.update(l_dense.item())
                
                l_fore_depth = criterion_fore_depth(pred, depth, mask=attention_mask.to(torch.bool), interpolate=True)
                print(f'The avg loss is {l_fore_depth.item()}')
                losses_fore_depth.update(l_fore_depth.item())
                        
                pred_process = pred.clone() # for projection


                pred = pred.detach().cpu().numpy()
                pred[pred < config.DATASET.MIN_DEPTH] = config.DATASET.MIN_DEPTH
                pred[pred > config.DATASET.MAX_DEPTH] = config.DATASET.MAX_DEPTH
                pred[np.isinf(pred)] = config.DATASET.MAX_DEPTH
                pred[np.isnan(pred)] = config.DATASET.MIN_DEPTH

                # depth_gt = depth.cpu().numpy()
                # valid_mask = np.logical_and(depth_gt > config.DATASET.MIN_DEPTH, depth_gt < config.DATASET.MAX_DEPTH)

                # 1. get the mask to do the filtering as "number"
                fitted_mask = F.interpolate(mask_prob, scale_factor=0.5)
                # filter mask
                filter_gradient_mask = generator_gradient(pred_process.clone())
                # import pdb; pdb.set_trace()
                # convolution is 2*P = F_size -1 for same 
                kernel = torch.ones(5,5).to(device)
                mask = (fitted_mask > 0.5).int()
                # erosion mask
                erode_mask = erode(mask,kernel.detach()) # erode error
                # import pdb; pdb.set_trace()
                filter_mask = erode_mask * filter_gradient_mask

                # for hm_idx in range(number_joints):
                #     hm_sampling = heatmap[:,hm_idx:hm_idx + 1,:,:]
                #     hm_max = torch.max(hm_sampling)
                #     choice_generator = 2 * hm_max * torch.rand(hm_sampling.shape).to(device)
                #     hm_sampling_mask = hm_sampling > choice_generator
                #     filter_mask.append(erode_mask * filter_gradient_mask * hm_sampling_mask)

                ## for global heatmap sampling
                # hm_sampling,_ = torch.max(heatmap, dim = 1, keepdims=True)
                # hm_max = torch.max(hm_sampling)
                # # get one sampling mask
                # choice_generator = 2 * hm_max * torch.rand(hm_sampling.shape).to(device)
                # hm_sampling_mask = hm_sampling > choice_generator
                # filter_mask = erode_mask * filter_gradient_mask * hm_sampling_mask


                # if all the mask is zero, then it is no mean to sample
                # import pdb; pdb.set_trace()
                # unproject the depth prediction (pred_process)
                
                # TODO : 
                total_points = points_extractor(pred_process, filter_mask, K_matrix, M_matrix,diff, trans_matrix, total_points, feature_out.clone()) # 

                # import pdb;pdb.set_trace()
                depth = F.interpolate(depth, scale_factor=0.5)
                gt_total_points = points_extractor(depth.float(), filter_mask, K_matrix, M_matrix,diff, trans_matrix, gt_total_points, feature_out.clone())
                
                ## below is the vis part

                # # vis the first batch
                # vis_depth = pred[0,0,...] 
                # # changed to ground truth
                # # vis_depth = depth_gt[0,0,...]
                # fitted_mask = F.interpolate(mask_prob, scale_factor=0.5)
                # temp_mask = fitted_mask[0,0,...].detach().cpu().numpy()

                # vis_mask = (temp_mask>0.5).astype(np.uint8)
                # # erode the mask
                # kernel = np.ones((5,5),np.uint8)
                # # kernel = CV2.getStructuringElement(CV2.MORPH_ELLIPSE,(5,5))

                # vis_mask = cv2.erode(vis_mask,kernel,iterations = 1) # no erosion

                # # get the gradient graph
                # scharrx = cv2.Sobel(vis_depth, cv2.CV_64F, 1, 0, ksize=-1)
                # scharry = cv2.Sobel(vis_depth, cv2.CV_64F, 0, 1, ksize=-1)

                # scharrxy = np.sqrt(scharrx*scharrx + scharry*scharry)

                # # plt.imshow(scharrxy, cmap='gray')
                # # plt.savefig(f'debug_pics/debug_depthgrad_{view}.png')

                # filter_depth = (scharrxy < 1).astype(np.uint8) # for filter

                # plt.imshow(filter_depth)
                # plt.savefig(f'debug_pics/debug_filter_{view}.png')
                # # import pdb; pdb.set_trace()
                

                # vis_sampling_mask = hm_sampling_mask[0,0,...].detach().cpu().numpy()
                
                # plt.imshow(vis_sampling_mask)
                # plt.savefig(f'debug_pics/debug_sampling_mask_{view}.png')
                # # print(hm_sampling.shape)
                # # vis_hm = hm_gt[0,0,...].detach().cpu().numpy()
                # # plt.imshow(vis_hm)
                # # plt.savefig(f'debug_pics/debug_gt_hm_{view}.png')
                # _ , point_3D = unprojectPoints(vis_depth,K_matrix[0],M_matrix[0],diff[0])
                # point_panoptic = trans_matrix[0] @ point_3D
                # point_panoptic = point_panoptic[:3,:].transpose()
                # filter_mask = vis_mask * filter_depth * vis_sampling_mask # TODO: 
                # # mask_filt = vis_mask.reshape(-1)
                # mask_filt = filter_mask.reshape(-1)

                # # get the gt points
                # filter_points = point_panoptic[np.where(mask_filt==1)] # select the feature at the same time
                # total_points.append(filter_points)
                
                # # with open(f'points_debug/point_whole_{view}.pkl','wb') as dfile:
                # #     pickle.dump(point_panoptic,dfile)

                # with open(f'points_debug/point_{view}.pkl','wb') as dfile:
                #     pickle.dump(filter_points,dfile)

                # plt.imshow(vis_depth, cmap='magma_r')
                # plt.savefig(f'debug_pics/debug_depth_{view}.png')
                # plt.imshow(vis_mask)
                # plt.savefig(f'debug_pics/debug_mask_{view}.png')

                # # pcd = o3d.geometry.PointCloud()
                # # pcd.points = o3d.utility.Vector3dVector(filter_points.copy() / 100.0)
                # # vis = o3d.visualization.Visualizer()
                # # vis.create_window()
                # # vis.add_geometry(pcd)
                # # vis.update_geometry(pcd)
                # # vis.poll_events()
                # # vis.update_renderer()
                # # vis.capture_screen_image('debug_points.png')
                # # vis.destroy_window()
                
                # # o3d.visualization.draw_geometries([pcd]) 
             
            # for hm_idx in range(number_joints):
            #     for b in range(batch_num):
            #         total_points[hm_idx][b] = torch.cat(total_points[hm_idx][b],dim=0)

            for b in range(batch_num):
                total_points[b] = torch.cat(total_points[b],dim=0)
                gt_total_points[b] = torch.cat(gt_total_points[b],dim=0)

            # 
            gt_test_points = gt_total_points[0][:,:3].detach().cpu().numpy()

            # existing the 0 joints sampling
            ###### get in the pointcloud input mode
            # for hm_idx in range(number_joints):
            #     batch_total_points = total_points[hm_idx]
            #     max_points_num = 0
            #     for b in range(batch_num):
            #         s_batch_points = batch_total_points[b]
            #         s_num = s_batch_points.shape[0]
            #         if s_num >= max_points_num:
            #             max_points_num = s_num
            #     # assure the max_points_num
            #     # import pdb; pdb.set_trace()
            #     for b in range(batch_num):
            #         s_batch_points = batch_total_points[b]
            #         s_num = s_batch_points.shape[0]
            #         if s_num == max_points_num:
            #             total_points[hm_idx][b] = s_batch_points.unsqueeze(0)
            #             continue
            #         offset = max_points_num - s_num
            #         fill_indx = torch.randint(s_num, (offset, ))
            #         fill_tensor = s_batch_points[fill_indx,:] # 索引可能不够了
            #         new_tensor = torch.cat([s_batch_points,fill_tensor],dim=0)
            #         total_points[hm_idx][b] = new_tensor.unsqueeze(0)
            #     # import pdb; pdb.set_trace()
            #     total_points[hm_idx] = torch.cat(total_points[hm_idx],dim=0)

                # ad
            batch_total_points = total_points
            max_points_num = 0
            for b in range(batch_num):
                s_batch_points = batch_total_points[b]
                s_num = s_batch_points.shape[0]
                if s_num >= max_points_num:
                    max_points_num = s_num
            num_max_points = 16384
            if max_points_num > num_max_points:
                for b in range(batch_num):
                    s_batch_points = batch_total_points[b]
                    s_num = s_batch_points.shape[0]
                    if s_num <= num_max_points:
                        offset = num_max_points - s_num
                        if s_num == 0:  # 其他batch 有值，该batch 在该关节点从处无值
                            fill_tensor = torch.zeros((max_points_num, 35)).to(torch.device('cuda'))
                        else:
                            fill_indx = torch.randint(s_num, (offset, ))
                            fill_tensor = s_batch_points[fill_indx,:] # 索引可能不够了
                        new_tensor = torch.cat([s_batch_points,fill_tensor],dim=0) # points must be float tensor
                        total_points[b] = new_tensor.unsqueeze(0)
                    else:
                        sample_idx = torch.randperm(s_num)[:num_max_points]
                        new_tensor = s_batch_points[sample_idx,:]
                        total_points[b] = new_tensor.unsqueeze(0)
            else:
                for b in range(batch_num):
                    s_batch_points = batch_total_points[b]
                    s_num = s_batch_points.shape[0]
                    if s_num == max_points_num:
                        total_points[b] = s_batch_points.unsqueeze(0)
                        continue
                    offset = max_points_num - s_num
                    if s_num == 0:  # 其他batch 有值，该batch 在该关节点从处无值
                        fill_tensor = torch.zeros((max_points_num, 35)).to(torch.device('cuda'))
                    else:
                        fill_indx = torch.randint(s_num, (offset, ))
                        fill_tensor = s_batch_points[fill_indx,:] # 索引可能不够了
                    new_tensor = torch.cat([s_batch_points,fill_tensor],dim=0) # points must be float tensor
                    total_points[b] = new_tensor.unsqueeze(0)

            total_points = torch.cat(total_points,dim=0)

            # total_points = np.concatenate(total_points, axis=0)
            test_points = total_points[0,:,:3].detach().cpu().numpy()

            with open(f'points_debug/point_total.pkl','wb') as dfile:
                pickle.dump(test_points,dfile)
            with open(f'points_debug/gt_point_total.pkl','wb') as dfile:
                pickle.dump(gt_test_points,dfile)
            
            import pdb; pdb.set_trace()

                    
            # batch_time.update(time.time() - end)
            # end = time.time()
            # if i % config.PRINT_FREQ == 0 or i == len(loader) - 1:
            #     gpu_memory_usage = torch.cuda.memory_allocated(0)
            #     msg = 'Test: [{0}/{1}]\t' \
            #           'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
            #           'Speed: {speed:.1f} samples/s\t' \
            #           'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
            #           'Memory {memory:.1f}'.format(
            #             i, len(loader), batch_time=batch_time,
            #             speed=image.size(0) / batch_time.val,
            #             data_time=data_time, memory=gpu_memory_usage)
            #     logger.info(msg)

    # metrics = metrics_c.get_value()
    # msg = 'a1: {aps_25:.4f}\ta2: {aps_50:.4f}\ta3: {aps_75:.4f}\t' \
    #         'abs_rel: {aps_100:.4f}\trmse: {aps_125:.4f}\tlog10: {aps_150:.4f}\t' \
    #         'rmse_log: {recall:.4f}\tsq_rel: {mpjpe:.3f}'.format(
    #         aps_25=metrics['a1'], aps_50=metrics['a2'], aps_75=metrics['a3'], aps_100=metrics['abs_rel'],
    #         aps_125=metrics['rmse'], aps_150=metrics['log_10'], recall=metrics['rmse_log'], mpjpe=metrics['sq_rel']
    #         )
    # logger.info(msg)
    # return None



def unprojectPoints(depth, K, M, diff):
    H,W = depth.shape
    #import pdb; pdb.set_trace()
    x_cor, y_cor = np.meshgrid(range(W), range(H))
    x_cor = x_cor + 71 # for croping 
    # do the transformation 
    orig_H = 1080
    orig_W = 1920
    x_cor = x_cor * (orig_W / (480)) # TODO: this is for croping size
    y_cor = y_cor * (orig_H / H) # / the value of orig resize size
    x_cor = x_cor.reshape(-1,1)
    y_cor = y_cor.reshape(-1,1)

    cor_2d = np.concatenate([x_cor,y_cor,np.ones([x_cor.shape[0],1])],axis=-1).transpose()
    K_depth = K
    norm_2d = np.linalg.pinv(K_depth) @ cor_2d
    norm_2d = norm_2d.transpose()
    x_cor_depth = norm_2d[:,0:1]
    x_cor_bak = x_cor_depth.copy()
    y_cor_depth = norm_2d[:,1:2]
    y_cor_bak = y_cor_depth.copy()
    K_diff = diff
    # undistortion
    for _ in range(5):
        r2 = x_cor_depth * x_cor_depth + y_cor_depth * y_cor_depth
        icdist = (1 + ((K_diff[7]*r2 + K_diff[6])*r2 + K_diff[5])*r2) / (1 + ((K_diff[4]*r2 + K_diff[1])*r2 + K_diff[0])*r2)
        deltaX = 2*K_diff[2] *x_cor_depth *y_cor_depth + K_diff[3]*(r2 + 2*x_cor_depth * x_cor_depth)+  K_diff[8]*r2+K_diff[9]* r2 *r2
        deltaY = K_diff[2]*(r2 + 2*y_cor_depth *y_cor_depth) + 2*K_diff[3]*x_cor_depth *y_cor_depth+ K_diff[10] * r2 + K_diff[11]* r2 *r2

        x_cor_depth = (x_cor_bak - deltaX) *icdist
        y_cor_depth = (y_cor_bak - deltaY) *icdist

    depth_proc = depth.reshape(-1,1) 
    x_cor_depth = x_cor_depth * depth_proc
    y_cor_depth = y_cor_depth * depth_proc
    depth_cam = np.concatenate([x_cor_depth,y_cor_depth,depth_proc,np.ones(x_cor_depth.shape)],axis=-1)
    M_depth = M
    point_3D = np.linalg.pinv(M_depth) @ depth_cam.transpose()
    point_3d = point_3D[:3,:].transpose()

    return point_3d, point_3D


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def nn_distance(pc1, pc2, l1smooth=False, delta=1.0, l1=False):
    """
    Input:
        pc1: (B,N,C) torch tensor
        pc2: (B,M,C) torch tensor
        l1smooth: bool, whether to use l1smooth loss
        delta: scalar, the delta used in l1smooth loss
    Output:
        dist1: (B,N) torch float32 tensor
        idx1: (B,N) torch int64 tensor
        dist2: (B,M) torch float32 tensor
        idx2: (B,M) torch int64 tensor
    """

    N = pc1.shape[1] 
    M = pc2.shape[1]

    pc1_expand_tile = pc1.unsqueeze(2).repeat(1,1,M,1)
    pc2_expand_tile = pc2.unsqueeze(1).repeat(1,N,1,1)
    pc_diff = pc1_expand_tile - pc2_expand_tile
    
    pc_dist = torch.sum(pc_diff**2, dim=-1) # (B,N,M)
    
    # import pdb; pdb.set_trace()
    dist1, idx1 = torch.min(pc_dist, dim=2) # (B,N)  每个推测点距离所有真值点最近的 距离 和真值indx
    dist2, idx2 = torch.min(pc_dist, dim=1) # (B,M)  每个真值点距离所有推测点最近的 推测Index 和 距离
    return dist1, idx1, dist2, idx2

# tensor2img tools
def tensor2im(input_image, imtype=np.uint8):
    """"将tensor的数据类型转成numpy类型，并反归一化.

    Parameters:
        input_image (tensor) --  输入的图像tensor数组
        imtype (type)        --  转换后的numpy的数据类型
    """
    mean = [0.485,0.456,0.406] #dataLoader中设置的mean参数
    std = [0.229,0.224,0.225]  #dataLoader中设置的std参数
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor): #如果传入的图片类型为torch.Tensor，则读取其数据进行下面的处理
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.detach().cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        for i in range(len(mean)): #反标准化
            image_numpy[i] = image_numpy[i] * std[i] + mean[i]
        image_numpy = image_numpy * 255 #反ToTensor(),从[0,1]转为[0,255]
        image_numpy = np.transpose(image_numpy, (1, 2, 0))  # 从(channels, height, width)变为(height, width, channels)
    else:  # 如果传入的是numpy数组,则不做处理
        image_numpy = input_image
    return image_numpy.astype(imtype)





class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
