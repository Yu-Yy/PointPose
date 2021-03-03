import torch
import torch.nn as nn
import torch.nn.functional as F

BN_MOMENTUM = 0.1

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PoseMaskEstimation(nn.Module):
    def __init__(self,cfg, basicM_num = 1):
        super(PoseMaskEstimation, self).__init__()
        self.pose_l = self.__make_pose__(basicM_num, cfg.MODEL_EXTRA.STAGE4.NUM_CHANNELS[0], cfg.NETWORK.NUM_JOINTS)
        self.mask_l = self.__make_mask__(basicM_num, cfg.MODEL_EXTRA.STAGE4.NUM_CHANNELS[0])

    def __make_pose__(self, basicM_num, inplane, num_joints):
        layers = []
        for _ in range(basicM_num):
            layers.append(nn.Sequential(
                BasicBlock(inplane, inplane),
            ))
        transition = nn.Sequential(*layers)
        final_layer = nn.Sequential(nn.Conv2d(
                in_channels=inplane,
                out_channels=num_joints,
                kernel_size=3,
                stride=1,
                padding=1
            ), nn.ReLU(inplace=True))
        return nn.Sequential(transition,final_layer)
    
    def __make_mask__(self, basicM_num, inplane):
        layers = []
        for _ in range(basicM_num):
            layers.append(nn.Sequential(
                BasicBlock(inplane, inplane),
            ))
        transition = nn.Sequential(*layers)
        final_layer = nn.Sequential(nn.Conv2d(
                in_channels=inplane,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1
            ),nn.Upsample(scale_factor=(2,2), mode='bilinear',align_corners=True) ,nn.Sigmoid())
        return nn.Sequential(transition,final_layer)

    def forward(self, feature):
        pose = self.pose_l(feature)
        mask = self.mask_l(feature)
        return pose, mask
        

        