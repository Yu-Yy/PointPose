import torch
import torch.nn as nn
import torch.nn.functional as F

# from .miniViT import mViT
from models import pose_higher_hrnet
from models.depth_header import DepthEstimation
from models.PM_header import PoseMaskEstimation


class HrnetAdaptiveBins(nn.Module):
    def __init__(self, cfg, backend, is_train = True):
        super(HrnetAdaptiveBins, self).__init__()
        # self.encoder = Encoder(backend)
        self.backbone = backend # extracting features        
        # depth header
        self.depth_header = eval('DepthEstimation.build')( # hrnet_adabins.build
                        cfg, cfg.BINS)
        self.PM_header = PoseMaskEstimation(cfg, basicM_num=1)

    def forward(self, x, **kwargs):
        # unet_out = self.decoder(self.encoder(x), **kwargs) # 128通道特征输出，batch 128 h/2 w/2
        # import pdb; pdb.set_trace()
        hrnet_out = self.backbone(x) # 维度输出有问题
        # FEATURE output channel fixed
        bin_edges, pred = self.depth_header(hrnet_out)
        # mask and poseH output
        heatmap, mask_prob = self.PM_header(hrnet_out)

        return bin_edges, pred, heatmap, mask_prob

    @classmethod
    def build(cls, cfg, is_train, **kwargs):
        # Building Encoder-Decoder model
        print('Building Hrnet_adabin model..', end='')
        backbone = eval(cfg.BACKBONE_MODEL + '.get_pose_net')(cfg, is_train=is_train)
        m = cls(cfg, backbone, is_train=is_train)
        print('Done.')
        return m

