import torch
import torch.nn as nn
from pytorch3d.loss import chamfer_distance # temperally commit
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


class SILogLoss(nn.Module):  # Main loss function used in AdaBins paper
    def __init__(self):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'

    def forward(self, input, uncertainty ,target, mask=None, interpolate=True):
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)
            uncertainty = nn.functional.interpolate(uncertainty, target.shape[-2:], mode='bilinear', align_corners=True)
        if mask is not None:
            input = input[mask]
            target = target[mask]
            uncertainty = uncertainty[mask]
        g = torch.log(input) - torch.log(target)
        # consider the uncertainty
        g_un = g * torch.exp(-uncertainty) + uncertainty # augment the punishment

        # n, c, h, w = g.shape
        # norm = 1/(h*w)
        # Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2

        # Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        Dg = torch.var(g) + torch.pow(torch.mean(g_un), 2) #0.15 * 
        return 10 * torch.sqrt(Dg)


class SILogLoss_grad(nn.Module):  # Main loss function used in AdaBins paper
    def __init__(self):
        super(SILogLoss_grad, self).__init__()
        device=torch.device('cuda')
        self.x_kernel = torch.FloatTensor([[-3, 0.01, 3],[-10,0.01,10],[-3,0.01,3]]).to(device).unsqueeze(0).unsqueeze(0)
        self.y_kernel = torch.FloatTensor([[-3, -10, -3],[0.01,0.01,0.01],[3,10,3]]).to(device).unsqueeze(0).unsqueeze(0)

        self.name = 'SILog'

    def forward(self, input, target, mask=None, interpolate=True):
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True) 
        # calculate the gradient
        grad_x_pred = F.conv2d(input,self.x_kernel,padding=1)
        grad_y_pred = F.conv2d(input,self.y_kernel,padding=1)
        xy_pred = torch.sqrt(torch.pow(grad_x_pred, 2) + torch.pow(grad_y_pred, 2))
        grad_x_gt = F.conv2d(target,self.x_kernel,padding=1)
        grad_y_gt = F.conv2d(target,self.y_kernel,padding=1)
        xy_gt = torch.sqrt(torch.pow(grad_x_gt, 2) + torch.pow(grad_y_gt, 2))
        
        # import pdb; pdb.set_trace()
        if mask is not None:
            xy_pred = xy_pred[mask]
            xy_gt = xy_gt[mask]
        # print(xy_gt)
        g = abs(xy_pred - xy_gt)  # bbox has low gradient
        # g = xy_pred
        # n, c, h, w = g.shape
        # norm = 1/(h*w)
        # Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2
        Dg = torch.var(g) + torch.pow(torch.mean(g), 2) #0.15 *
        return 10 * torch.sqrt(Dg)




class BinsChamferLoss(nn.Module):  # Bin centers regularizer used in AdaBins paper
    def __init__(self):
        super().__init__()
        self.name = "ChamferLoss"

    def forward(self, bins, target_depth_maps):
        bin_centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
        n, p = bin_centers.shape
        device = bin_centers.device
        input_points = bin_centers.view(n, p, 1)  # .shape = n, p, 1
        # n, c, h, w = target_depth_maps.shape
        
        target_points = target_depth_maps.flatten(1)  # n, hwc
        mask = target_points.ge(1e-3)  # only valid ground truth points
        target_points = [p[m] for p, m in zip(target_points, mask)]
        target_lengths = torch.Tensor([len(t) for t in target_points]).long().to(target_depth_maps.device)
        target_points = pad_sequence(target_points, batch_first=True).unsqueeze(2)  # .shape = n, T, 1

        loss, _ = chamfer_distance(x=input_points, y=target_points, y_lengths=target_lengths)
        return loss

class PerJointMSELoss(nn.Module):
    def __init__(self):
        super(PerJointMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, output, target, gt_mask, use_target_weight = False, target_weight=None, sigma = 3):
        # add one mask loss
        fitted_mask = F.interpolate(gt_mask, scale_factor=0.5)
        loss_attention = ((output - target)**2) * fitted_mask[:, 0:1, :, :].expand_as(output)# (reshape ) # mask pred batch,n_j,h,w
        loss_attention = loss_attention.mean(dim=3).mean(dim=2).mean(dim=1).mean(dim=0) # be one number
        if use_target_weight:
            batch_size = output.size(0)
            num_joints = output.size(1)

            heatmap_pred = output.reshape((batch_size, num_joints, -1))
            heatmap_gt = target.reshape((batch_size, num_joints, -1))
            loss_global = self.criterion(heatmap_pred.mul(target_weight), heatmap_gt.mul(target_weight))
        else:
            loss_global = self.criterion(output, target)

        # loss total num
        # diff_abs_sum = torch.abs(torch.sum(output) - torch.sum(target))
        # number_loss = 1 - torch.exp( - diff_abs_sum/(2 * (sigma ** 2)))

        # import pdb;pdb.set_trace()
        loss = 0.1 * loss_global + 0.9 * loss_attention # pay more attention to the hm lighted # It is affected by the view number ?
        # loss = loss_global + number_loss
        return loss

class CrossEntropyMaskLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyMaskLoss,self).__init__()
        self.criterion = nn.CrossEntropyLoss() # default reduction 'mean'
    def forward(self, mask_pro, gt_mask):
        # in batch crossentropy
        mask_pro = mask_pro.reshape(-1,1)
        mask_prob_0 = 1 - mask_pro
        input = torch.cat([mask_prob_0,mask_pro],axis=1)
        gt_mask = gt_mask.reshape(-1) # 
        loss = self.criterion(input, gt_mask)
        return loss

class foreground_depth_loss(nn.Module):
    def __init__(self):
        super(foreground_depth_loss, self).__init__()
        self.name = 'loss_test'

    def forward(self, input, target, mask=None, interpolate=True):
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)

        if mask is not None:
            input = input[mask]
            target = target[mask]
        g = input - target
        # n, c, h, w = g.shape
        # norm = 1/(h*w)
        # Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2

        Dg = torch.pow(g, 2)
        return torch.mean(torch.sqrt(Dg))