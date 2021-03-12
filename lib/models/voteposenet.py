# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Deep hough voting network for 3D object detection in point clouds.

Author: Charles R. Qi and Or Litany
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.dirname(BASE_DIR)
# sys.path.append(BASE_DIR)
from models.votepose_utils import VotePoseBackbone, VotePoseVoting, VotePoseProposal, get_loss

#from dump_helper import dump_results
#from loss_helper import get_loss
import time

class VotePoseNet(nn.Module):
    r"""
        A deep neural network for 3D object detection with end-to-end optimizable hough voting.  # votenet 

        Parameters
        ----------
        num_class: int
            Number of semantics classes to predict over -- size of softmax classifier
        num_heading_bin: int
        num_size_cluster: int
        input_feature_dim: (default: 0)
            Input dim in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        num_proposal: int (default: 128)
            Number of proposals/detections generated from the network. Each proposal is a 3D OBB with a semantic class.
        vote_factor: (default: 1)
            Number of votes generated from each seed point.
    """

    def __init__(self, input_feature_dim=0, num_proposal=128, vote_factor=1, sampling='vote_fps'):
        super().__init__()

        #self.num_class = num_class
        #self.num_heading_bin = num_heading_bin
        #self.num_size_cluster = num_size_cluster
        #self.mean_size_arr = mean_size_arr
        #assert(mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        # self.sampling=sampling # using ?

        # Backbone point feature learning
        self.backbone_net = VotePoseBackbone(input_feature_dim=self.input_feature_dim)

        # Hough voting
        self.vgen = VotePoseVoting(self.vote_factor, 128) # 128

        # Vote aggregation and detection
        self.pnet = VotePoseProposal(num_proposal, sampling, seed_feat_dim=128) # 128 dim output

    def forward(self, inputs):
        """ Forward pass of the network

        Args:
            inputs: tensor.float

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """
        end_points = {}
        batch_size = inputs.shape[0]

        end_points = self.backbone_net(inputs.float(), end_points) # changed into float tensor
                
        # --------- HOUGH VOTING ---------
        xyz = end_points['fp1_xyz']
        features = end_points['fp1_features']
        end_points['seed_inds'] = end_points['fp1_inds']
        end_points['seed_xyz'] = xyz
        end_points['seed_features'] = features
        
        xyz, features = self.vgen(xyz, features)  
        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))
        end_points['vote_xyz'] = xyz # one suprervise 
        end_points['vote_features'] = features
        #print(features.shape)
        end_points = self.pnet(xyz, features, end_points)

        return end_points


if __name__=='__main__':
    # sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
    #from sunrgbd_detection_dataset import SunrgbdDetectionVotesDataset, DC
    #from loss_helper import get_loss

    # Define model
    model = VotePoseNet(32).cuda()
    
    try:
        # Define dataset
        TRAIN_DATASET = SunrgbdDetectionVotesDataset('train', num_points=20000, use_v1=True)

        # Model forward pass
        sample = TRAIN_DATASET[5]
        inputs = {'point_clouds': torch.from_numpy(sample['point_clouds']).unsqueeze(0).cuda()}
    except:
        print('Dataset has not been prepared. Use a random sample.')
        inputs = torch.rand((19,8,3800,35)).cuda()  #.unsqueeze(0)
    t = time.time()
    for i in range(19):
        end_points = model(inputs[i])
    # import pdb; pdb.set_trace()
        gt_points = torch.rand((8,4,3)).cuda()
        loss, end_points = get_loss(end_points, gt_points)
    # end_points = model(inputs)
    batch_time = time.time() - t
    print(f'{batch_time}s')
    speed = inputs.shape[1] / batch_time
    print(f'{speed} samp/s')
    
    '''
    for key in end_points:
        print(key, end_points[key])
    '''
    # try:
    #     # Compute loss
    #     for key in sample:
    #         end_points[key] = torch.from_numpy(sample[key]).unsqueeze(0).cuda()
    #     loss, end_points = get_loss(end_points, DC)
    #     print('loss', loss)
    #     end_points['point_clouds'] = inputs['point_clouds']
    #     end_points['pred_mask'] = np.ones((1,128))
    #     # dump_results(end_points, 'tmp', DC)
    # except:

    #     print('Dataset has not been prepared. Skip loss and dump.')

    # testing loss 
    
