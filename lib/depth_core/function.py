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


# from models.hrnet_adabins import HrnetAdaptiveBins
import matplotlib.pyplot as plt
import open3d as o3d
import cv2

logger = logging.getLogger(__name__)


def train_depth(config, model, optimizer, loader, epoch, output_dir, writer_dict, device=torch.device('cuda'), dtype=torch.float):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_depth = AverageMeter()
    losses_hm = AverageMeter()
    losses_mask = AverageMeter()
    losses_fore_depth = AverageMeter()
    
    criterion_hm = PerJointMSELoss().cuda()
    criterion_dense = SILogLoss().cuda()
    criterion_dense_attention = SILogLoss().cuda()
    criterion_chamfer = BinsChamferLoss().cuda()
    criterion_mask = CrossEntropyMaskLoss().cuda()
    criterion_grad = SILogLoss_grad().cuda()
    criterion_fore_depth = foreground_depth_loss().cuda()

    model.train()

    end = time.time()
    # multi_loader_training shelf and cmu
    for i, batch in enumerate(loader): # 
        if len(batch) == 0:
            continue
        image, depth, mask_gt, hm_gt, wt_gt = batch
        image = image.to(device)
        depth = depth.to(device)
        mask_gt = mask_gt.to(device)
        hm_gt = hm_gt.to(device)
        wt_gt = wt_gt.to(device)

        # import pdb; pdb.set_trace()
        bin_edges, pred, heatmap, mask_prob = model(image) # depth is in half resolution

        # depth loss
        depth_mask = depth > config.DATASET.MIN_DEPTH
        attention_mask = depth_mask * mask_gt
        l_dense = criterion_dense(pred, depth, mask=depth_mask.to(torch.bool), interpolate=True)
        # add one heatmap mask to pay more attention to the keypoint position


        l_dense_attention = criterion_dense_attention(pred, depth, mask=attention_mask.to(torch.bool), interpolate=True)
        l_chamfer = criterion_chamfer(bin_edges, depth.float())
        l_grad = criterion_grad(pred, depth.detach().float(), mask=attention_mask.to(torch.bool), interpolate=True) # do not add this loss 
        # print(l_grad)

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

        loss = loss_depth + 50 * loss_hm + loss_mask * 5
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
                  'Loss_mask: {loss_cord.val:.6f} ({loss_cord.avg:.6f})\t' \
                  'Loss_fore_depth: {loss_fore_depth.val:.6f}({loss_fore_depth.avg:.6f})\t' \
                  'Memory {memory:.1f}'.format(
                    epoch, i, len(loader), batch_time=batch_time,
                    speed=image.size(0) / batch_time.val,
                    data_time=data_time, loss=losses, loss_2d=losses_depth, loss_3d=losses_hm, loss_fore_depth = losses_fore_depth,
                    loss_cord=losses_mask, memory=gpu_memory_usage)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_loss_mask', losses_mask.val, global_steps)
            writer.add_scalar('train_loss_hm', losses_hm.val, global_steps)
            writer.add_scalar('train_loss_depth', losses_depth.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            # debug file
            if i % 1500 == 0:
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
                vis_hm = heatmap[0,0,...].detach().cpu().numpy()
                folder_name = os.path.join(output_dir, 'debug_train_pics')
                depth_folder = os.path.join(folder_name, 'depth')
                hm_folder = os.path.join(folder_name,'heatmap')
                mask_folder = os.path.join(folder_name,'mask')

                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)
                if not os.path.exists(depth_folder):
                    os.makedirs(depth_folder)
                if not os.path.exists(hm_folder):
                    os.makedirs(hm_folder)
                if not os.path.exists(mask_folder):
                    os.makedirs(mask_folder)
                plt.imshow(vis_hm)
                plt.savefig(os.path.join(hm_folder, f'hm_{epoch}_i_{i}.jpg'))
                plt.imshow(vis_depth,cmap='magma_r')
                plt.savefig(os.path.join(depth_folder, f'depth_{epoch}_i_{i}.jpg'))
                plt.imshow(vis_mask)
                plt.savefig(os.path.join(mask_folder, f'mask_{epoch}_i_{i}.jpg'))
                





def validate_depth(config, model, loader, output_dir, epoch=0, vali=False, device=torch.device('cuda')):
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
    
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, batch in enumerate(loader):
            if len(batch) == 0:
                continue
            data_time.update(time.time() - end)
            image, depth, mask_gt, hm_gt, wt_gt = batch
            image = image.to(device)
            depth = depth.to(device)
            mask_gt = mask_gt.to(device)
            hm_gt = hm_gt.to(device)
            wt_gt = wt_gt.to(device)

            bin_edges, pred, heatmap, mask_prob = model(image)

            # vis_mask = (mask_prob > 0.5).astype(np.int)

            depth_mask = depth > config.DATASET.MIN_DEPTH
            attention_mask = depth_mask * mask_gt
            l_dense = criterion_dense(pred, depth, mask=depth_mask.to(torch.bool), interpolate=True)
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
                    folder_name = os.path.join(output_dir, 'debug_test_pics')
                    depth_folder = os.path.join(folder_name, 'depth')
                    hm_folder = os.path.join(folder_name,'heatmap')
                    mask_folder = os.path.join(folder_name,'mask')
                    if not os.path.exists(folder_name):
                        os.makedirs(folder_name)
                    if not os.path.exists(depth_folder):
                        os.makedirs(depth_folder)
                    if not os.path.exists(hm_folder):
                        os.makedirs(hm_folder)
                    if not os.path.exists(mask_folder):
                        os.makedirs(mask_folder)
                    plt.imshow(vis_hm)
                    plt.savefig(os.path.join(hm_folder, f'hm_{epoch}_i_{i}.jpg'))
                    plt.imshow(vis_depth,cmap='magma_r')
                    plt.savefig(os.path.join(depth_folder, f'depth_{epoch}_i_{i}.jpg'))
                    plt.imshow(vis_mask)
                    plt.savefig(os.path.join(mask_folder, f'mask_{epoch}_i_{i}.jpg'))

    metrics = metrics_c.get_value()
    msg = 'a1: {aps_25:.4f}\ta2: {aps_50:.4f}\ta3: {aps_75:.4f}\t' \
            'abs_rel: {aps_100:.4f}\trmse: {aps_125:.4f}\tlog10: {aps_150:.4f}\t' \
            'rmse_log: {recall:.4f}\tsq_rel: {mpjpe:.3f}'.format(
            aps_25=metrics['a1'], aps_50=metrics['a2'], aps_75=metrics['a3'], aps_100=metrics['abs_rel'],
            aps_125=metrics['rmse'], aps_150=metrics['log_10'], recall=metrics['rmse_log'], mpjpe=metrics['sq_rel']
            )
    logger.info(msg)
    return metrics

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
            output_img, output_depth, output_valid_mask,output_hm, output_weights, output_trans, output_K, output_M, output_diff, output_3d_pose = batch
            batch_num = output_img.shape[0]
            view_num = output_img.shape[1]
            # process in multiple view form
            total_points = [[] for _ in range(batch_num)]
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
                


                bin_edges, pred, heatmap, mask_prob, feature_out = model(image)

                # vis_mask = (mask_prob > 0.5).astype(np.int)

                depth_mask = depth > config.DATASET.MIN_DEPTH
                attention_mask = depth_mask * mask_gt
                
                
                l_dense = criterion_dense(pred, depth, mask=depth_mask.to(torch.bool), interpolate=True)
                losses_depth.update(l_dense.item())
                
                l_fore_depth = criterion_fore_depth(pred, depth, mask=attention_mask.to(torch.bool), interpolate=True)
                # print(f'The avg loss is {l_fore_depth.item()}')
                losses_fore_depth.update(l_fore_depth.item())
                        
                pred_process = pred.clone() # for projection


                pred = pred.detach().cpu().numpy()
                pred[pred < config.DATASET.MIN_DEPTH] = config.DATASET.MIN_DEPTH
                pred[pred > config.DATASET.MAX_DEPTH] = config.DATASET.MAX_DEPTH
                pred[np.isinf(pred)] = config.DATASET.MAX_DEPTH
                pred[np.isnan(pred)] = config.DATASET.MIN_DEPTH

                depth_gt = depth.cpu().numpy()
                valid_mask = np.logical_and(depth_gt > config.DATASET.MIN_DEPTH, depth_gt < config.DATASET.MAX_DEPTH)

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
                hm_sampling,_ = torch.max(heatmap, dim = 1, keepdims=True)
                hm_max = torch.max(hm_sampling)
                # get one sampling mask
                choice_generator = 2 * hm_max * torch.rand(hm_sampling.shape).to(device)
                hm_sampling_mask = hm_sampling > choice_generator
                # import pdb; pdb.set_trace()

                filter_mask = erode_mask * filter_gradient_mask * hm_sampling_mask
                # if all the mask is zero, then it is no mean to sample
                # import pdb; pdb.set_trace()
                # unproject the depth prediction (pred_process)
                
                # TODO : 
                total_points = points_extractor(pred_process, filter_mask, K_matrix, M_matrix,diff, trans_matrix, total_points, feature_out.clone()) # 


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

            for b in range(batch_num):
                total_points[b] = torch.cat(total_points[b],dim=0)

            ###### get in the pointcloud input mode

            # total_points = np.concatenate(total_points, axis=0)
            # test_points = total_points[0].detach().cpu().numpy()

            # with open(f'points_debug/point_total.pkl','wb') as dfile:
            #     pickle.dump(test_points,dfile)
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
