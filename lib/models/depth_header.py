import torch
import torch.nn as nn
import torch.nn.functional as F

from .miniViT import mViT
# from models import pose_higher_hrnet



class DepthEstimation(nn.Module):
    def __init__(self, cfg, n_bins=100, min_val=0.1, max_val=10, norm='linear'):
        super(DepthEstimation, self).__init__()
        self.num_classes = n_bins
        self.min_val = min_val
        self.max_val = max_val
        # self.encoder = Encoder(backend)
        # self.backbone = backend # extracting features
        self.num_query_seq = 128
        embedding_dim = 128
        self.ada_input_channels = cfg.MODEL_EXTRA.STAGE4.NUM_CHANNELS[0] # 32 channel feature
        self.adaptive_bins_layer = mViT(self.ada_input_channels, n_query_channels=self.num_query_seq, patch_size=16,  # 128 feature channel output # 
                                        dim_out=n_bins,
                                        embedding_dim=embedding_dim, norm=norm)

        # self.decoder = DecoderBN(num_classes=128)
        self.conv_out = nn.Sequential(nn.Conv2d(self.num_query_seq, n_bins, kernel_size=1, stride=1, padding=0), # 128 should be the query channel's number
                                      nn.Softmax(dim=1))
        # self.uncertainty_out = nn.Sequential(nn.Conv2d(self.ada_input_channels, embedding_dim, kernel_size=3, stride=1, padding=1),
        #                                     nn.LeakyReLU(),nn.Conv2d(embedding_dim, 1, kernel_size=3, stride=1, padding=1))


    def forward(self, feature, **kwargs):
        # unet_out = self.decoder(self.encoder(x), **kwargs) # 128通道特征输出，batch 128 h/2 w/2
        # import pdb; pdb.set_trace()
        # hrnet_out = self.backbone(x) # 维度输出有问题
        # FEATURE output channel fixed
        # import pdb; pdb.set_trace()
        bin_widths_normed, range_attention_maps = self.adaptive_bins_layer(feature) # attention map 相当于 每个像素点的 FC 
        out = self.conv_out(range_attention_maps)
        # depth_uncertainty = self.uncertainty_out(feature)
        # Post process
        # n, c, h, w = out.shape
        # hist = torch.sum(out.view(n, c, h * w), dim=2) / (h * w)  # not used for training

        bin_widths = (self.max_val - self.min_val) * bin_widths_normed  # .shape = N, dim_out
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_val)
        bin_edges = torch.cumsum(bin_widths, dim=1) # 累加值

        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1)
        # output the center as probalility

        pred = torch.sum(out * centers, dim=1, keepdim=True)
        return bin_edges, pred #, depth_uncertainty

    def get_1x_lr_params(self):  # lr/10 learning rate
        return self.encoder.parameters()

    def get_10x_lr_params(self):  # lr learning rate
        modules = [self.decoder, self.adaptive_bins_layer, self.conv_out]
        for m in modules:
            yield from m.parameters()

    @classmethod
    def build(cls, cfg, n_bins, **kwargs):
        m = cls(cfg, n_bins=n_bins, min_val = cfg.DATASET.MIN_DEPTH, max_val=cfg.DATASET.MAX_DEPTH,**kwargs)
        # print('Done.')
        return m