import base64
import math
import re
from io import BytesIO

import matplotlib.cm
import numpy as np
import torch
import torch.nn
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

##################################
# for getting the gredient map
class grad_generation(nn.Module):  # Main loss function used in AdaBins paper
    def __init__(self):
        super(grad_generation, self).__init__()
        device=torch.device('cuda')
        self.x_kernel = torch.FloatTensor([[-3, 0, 3],[-10,0,10],[-3,0,3]]).to(device).unsqueeze(0).unsqueeze(0)
        self.y_kernel = torch.FloatTensor([[-3, -10, -3],[0,0,0],[3,10,3]]).to(device).unsqueeze(0).unsqueeze(0)

    def forward(self, input):
        # calculate the gradient
        grad_x_pred = F.conv2d(input,self.x_kernel,padding=1)
        grad_y_pred = F.conv2d(input,self.y_kernel,padding=1)
        xy_pred = torch.sqrt(torch.pow(grad_x_pred, 2) + torch.pow(grad_y_pred, 2))
        grad_mask = (xy_pred < 1)
        return grad_mask

# for getting the erosion results
class erode_generation(nn.Module):
    def __init__(self):
        super(erode_generation, self).__init__()
        self.device = torch.device('cuda')
    def forward(self, input, kernel):
        # import pdb; pdb.set_trace()
        threshold = torch.sum(kernel)
        size = kernel.shape[0]
        padding = (size-1) / 2
        kernel_process = kernel.unsqueeze(0).unsqueeze(0)
        result = F.conv2d(input.float(),kernel_process,padding=int(padding))
        erode_result = (result >= threshold).int()
        return erode_result

# getting the unprojected 3d points
class get_3d_points(nn.Module):
    def __init__(self,orig_W,orig_H):
        super(get_3d_points, self).__init__()
        self.device = torch.device('cuda')
        self.orig_W = orig_W
        self.orig_H = orig_H

    def forward(self, pred_depth, filter_mask, K_matrix, M_matrix,diff, trans_matrix, total_points, feature_input):
        pred_depth = pred_depth.squeeze(1)
        _,C,_,_ = feature_input.shape
        B,H,W = pred_depth.shape
        num_joints = len(filter_mask)
        K_matrix = K_matrix.clone().detach()
        M_matrix = M_matrix.clone().detach()
        diff = diff.clone().detach()
        diff = diff.unsqueeze(2) # for multi
        trans_matrix = trans_matrix.clone().detach()
        x_cor, y_cor = torch.meshgrid(torch.arange(W), torch.arange(H))
        x_cor = x_cor + 71 # for cropping
        x_cor = x_cor.transpose(0,1)
        y_cor = y_cor.transpose(0,1)
        x_cor = x_cor.to(self.device)
        y_cor = y_cor.to(self.device)
        x_cor = x_cor * (self.orig_W / (480)) # TODO: this is for croping size!
        y_cor = y_cor * (self.orig_H / H)
        x_cor = x_cor.reshape(-1,1)
        y_cor = y_cor.reshape(-1,1)
        cor_2d = torch.cat([x_cor,y_cor,torch.ones(x_cor.shape[0],1).to(self.device)],dim=-1).transpose(0,1)
        cor_2d_batch = cor_2d.unsqueeze(0).repeat(B,1,1)
        norm_2d = torch.bmm(torch.inverse(K_matrix), cor_2d_batch.double())
        norm_2d = norm_2d.permute([0,2,1])
        x_cor_depth = norm_2d[...,0:1]
        x_cor_bak = x_cor_depth.clone().detach()
        y_cor_depth = norm_2d[...,1:2]
        y_cor_bak = y_cor_depth.clone().detach()
        # import pdb; pdb.set_trace()

        for _ in range(5):
            r2 = x_cor_depth * x_cor_depth + y_cor_depth * y_cor_depth
            icdist = (1 + ((diff[...,7:8,:]*r2 + diff[...,6:7,:])*r2 + diff[...,5:6,:])*r2) / (1 + ((diff[...,4:5,:]*r2 + diff[...,1:2,:])*r2 + diff[...,0:1,:])*r2)
            deltaX = 2*diff[...,2:3,:] *x_cor_depth *y_cor_depth + diff[...,3:4,:]*(r2 + 2*x_cor_depth * x_cor_depth)+  diff[...,8:9,:]*r2 + diff[...,9:10,:]* r2 *r2
            deltaY = diff[...,2:3,:]*(r2 + 2*y_cor_depth *y_cor_depth) + 2 * diff[...,3:4,:]*x_cor_depth *y_cor_depth+ diff[...,10:11,:] * r2 + diff[...,11:12,:]* r2 *r2

            x_cor_depth = (x_cor_bak - deltaX) *icdist
            y_cor_depth = (y_cor_bak - deltaY) *icdist
        
        depth_proc = pred_depth.reshape(B,-1,1)
        x_cor_depth = x_cor_depth * depth_proc
        y_cor_depth = y_cor_depth * depth_proc
        # depth_cam = np.concatenate([x_cor_depth,y_cor_depth,depth_proc,np.ones(x_cor_depth.shape)],axis=-1)
        depth_cam = torch.cat([x_cor_depth, y_cor_depth,depth_proc,torch.ones(x_cor_depth.shape).to(self.device)],dim=-1)

        point_3D = torch.bmm(torch.inverse(M_matrix), depth_cam.permute([0,2,1]))
        # point_3D = np.linalg.pinv(M_depth) @ depth_cam.transpose()
        # point_3d = point_3D[:3,:].transpose()
        point_panoptic = torch.bmm(trans_matrix, point_3D)
        point_panoptic = point_panoptic[:,:3,:].permute([0,2,1])
        
        # batch_points = []
        feature_input = feature_input.reshape(B,C,-1)
        feature_input = feature_input.permute([0,2,1])
        # import pdb; pdb.set_trace()

        for hm_idx in range(num_joints):
            mask_filt = filter_mask[hm_idx].reshape(B,-1,1)
            for b in range(B):
                batch_point_3d = point_panoptic[b]
                batch_feature = feature_input[b]
                i_idx,j_idx = torch.where(mask_filt[b] == 1)
                filter_point_b = batch_point_3d[i_idx]
                filter_feature_b = batch_feature[i_idx]
                # combine the feature part
                filter_pc = torch.cat([filter_point_b,filter_feature_b],dim = -1)
                total_points[hm_idx][b].append(filter_pc)
                # batch_points.append(filter_point_b.unsqueeze(0))

        return total_points



####################################
class RunningAverage:
    def __init__(self):
        self.avg = 0
        self.count = 0

    def append(self, value):
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1

    def get_value(self):
        return self.avg


def denormalize(x, device='cpu'):
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    return x * std + mean


class RunningAverageDict:
    def __init__(self):
        self._dict = None

    def update(self, new_dict):
        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = RunningAverage()

        for key, value in new_dict.items():
            self._dict[key].append(value)

    def get_value(self):
        return {key: value.get_value() for key, value in self._dict.items()}


def colorize(value, vmin=10, vmax=1000, cmap='magma_r'):
    value = value.cpu().numpy()[0, :, :]
    invalid_mask = value == -1

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.
    # squeeze last dim if it exists
    # value = value.squeeze(axis=0)
    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # (nxmx4)
    value[invalid_mask] = 255
    img = value[:, :, :3]

    #     return img.transpose((2, 0, 1))
    return img


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel)


##################################### Demo Utilities ############################################
def b64_to_pil(b64string):
    image_data = re.sub('^data:image/.+;base64,', '', b64string)
    # image = Image.open(cStringIO.StringIO(image_data))
    return Image.open(BytesIO(base64.b64decode(image_data)))


# Compute edge magnitudes
from scipy import ndimage


def edges(d):
    dx = ndimage.sobel(d, 0)  # horizontal derivative
    dy = ndimage.sobel(d, 1)  # vertical derivative
    return np.abs(dx) + np.abs(dy)


class PointCloudHelper():
    def __init__(self, width=640, height=480):
        self.xx, self.yy = self.worldCoords(width, height)

    def worldCoords(self, width=640, height=480):
        hfov_degrees, vfov_degrees = 57, 43
        hFov = math.radians(hfov_degrees)
        vFov = math.radians(vfov_degrees)
        cx, cy = width / 2, height / 2
        fx = width / (2 * math.tan(hFov / 2))
        fy = height / (2 * math.tan(vFov / 2))
        xx, yy = np.tile(range(width), height), np.repeat(range(height), width)
        xx = (xx - cx) / fx
        yy = (yy - cy) / fy
        return xx, yy

    def depth_to_points(self, depth):
        depth[edges(depth) > 0.3] = np.nan  # Hide depth edges
        length = depth.shape[0] * depth.shape[1]
        # depth[edges(depth) > 0.3] = 1e6  # Hide depth edges
        z = depth.reshape(length)

        return np.dstack((self.xx * z, self.yy * z, z)).reshape((length, 3))

#####################################################################################################
