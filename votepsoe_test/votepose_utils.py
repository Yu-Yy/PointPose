'''
    TODO: transfer this code into the libs/models
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.dirname(BASE_DIR)
# sys.path.append(ROOT_DIR)
# sys.path.append(os.path.join(ROOT_DIR, 'utils'))
# sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
# print(ROOT_DIR)
from utils.nn_distance import nn_distance

from pointnet2.pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule

class VotePoseBackbone(nn.Module):
    r"""
       Backbone network for point cloud feature learning.
       Based on Pointnet++ single-scale grouping network. 
        
       Parameters
       ----------
       input_feature_dim: int
            Number of input channels in the feature descriptor for each point.
            e.g. 3 for RGB.
    """
    def __init__(self, input_feature_dim=32):
        super().__init__()
        #print(input_feature_dim)
        self.sa1 = PointnetSAModuleVotes(
                npoint=256,
                radius=0.2,
                nsample=256, # ?? orig 1024  
                mlp=[input_feature_dim, 64, 64, 128],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa2 = PointnetSAModuleVotes(
                npoint=128,
                radius=0.4,
                nsample=64,
                mlp=[128, 64, 64, 128],
                use_xyz=True,
                normalize_xyz=True
            )
        
        self.fp1 = PointnetFPModule(mlp=[128+128,128,128])
        #self.fp2 = PointnetFPModule(mlp=[256+256,256,256])

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor, end_points=None):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_feature_dim) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)

            Returns
            ----------
            end_points: {XXX_xyz, XXX_features, XXX_inds}
                XXX_xyz: float32 Tensor of shape (B,K,3)
                XXX_features: float32 Tensor of shape (B,K,D)
                XXX-inds: int64 Tensor of shape (B,K) values in [0,N-1]
        """
        if not end_points: end_points = {}
        batch_size = pointcloud.shape[0]

        xyz, features = self._break_up_pc(pointcloud)

        # --------- 2 SET ABSTRACTION LAYERS ---------
        #print(self.sa1)
        xyz, features, fps_inds = self.sa1(xyz, features)
        end_points['sa1_inds'] = fps_inds
        end_points['sa1_xyz'] = xyz
        end_points['sa1_features'] = features
        
        xyz, features, fps_inds = self.sa2(xyz, features) # this fps_inds is just 0,1,...,1023
        end_points['sa2_inds'] = fps_inds
        end_points['sa2_xyz'] = xyz
        end_points['sa2_features'] = features
        
        
        # --------- FEATURE UPSAMPLING LAYERS --------
        features = self.fp1(end_points['sa1_xyz'], end_points['sa2_xyz'], end_points['sa1_features'], end_points['sa2_features'])
        #print(end_points['sa1_xyz'].shape, end_points['sa2_xyz'].shape, end_points['sa1_features'].shape, end_points['sa2_features'].shape)
        #print(features.shape)
        end_points['fp1_features'] = features
        end_points['fp1_xyz'] = end_points['sa1_xyz']
        num_seed = end_points['fp1_xyz'].shape[1]
        end_points['fp1_inds'] = end_points['sa1_inds'][:,0:num_seed] # indices among the entire input point clouds
        return end_points

class VotePoseVoting(nn.Module):
    def __init__(self, vote_factor, seed_feature_dim):
        """ Votes generation from seed point features.

        Args:
            vote_facotr: int
                number of votes generated from each seed point
            seed_feature_dim: int
                number of channels of seed point features
            vote_feature_dim: int
                number of channels of vote features
        """
        super().__init__()
        self.vote_factor = vote_factor
        self.in_dim = seed_feature_dim
        self.out_dim = self.in_dim # due to residual feature, in_dim has to be == out_dim
        self.conv1 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv3 = torch.nn.Conv1d(self.in_dim, (3+self.out_dim) * self.vote_factor, 1)
        self.bn1 = torch.nn.BatchNorm1d(self.in_dim)
        self.bn2 = torch.nn.BatchNorm1d(self.in_dim)
        
    def forward(self, seed_xyz, seed_features):
        """ Forward pass.

        Arguments:
            seed_xyz: (batch_size, num_seed, 3) Pytorch tensor
            seed_features: (batch_size, feature_dim, num_seed) Pytorch tensor
        Returns:
            vote_xyz: (batch_size, num_seed*vote_factor, 3)
            vote_features: (batch_size, vote_feature_dim, num_seed*vote_factor)
        """
        batch_size = seed_xyz.shape[0]
        num_seed = seed_xyz.shape[1]
        num_vote = num_seed*self.vote_factor
        net = F.relu(self.bn1(self.conv1(seed_features))) 
        net = F.relu(self.bn2(self.conv2(net))) 
        net = self.conv3(net) # (batch_size, (3+out_dim)*vote_factor, num_seed)
        #print(seed_xyz.shape, seed_features.shape)
        #print(net.shape)
        net = net.transpose(2,1).view(batch_size, num_seed, self.vote_factor, 3+self.out_dim)
        offset = net[:,:,:,0:3]
        vote_xyz = seed_xyz.unsqueeze(2) + offset
        vote_xyz = vote_xyz.contiguous().view(batch_size, num_vote, 3)
        
        residual_features = net[:,:,:,3:] # (batch_size, num_seed, vote_factor, out_dim)
        vote_features = seed_features.transpose(2,1).unsqueeze(2) + residual_features
        vote_features = vote_features.contiguous().view(batch_size, num_vote, self.out_dim)
        vote_features = vote_features.transpose(2,1).contiguous()
        
        return vote_xyz, vote_features

def decode_scores(net, end_points):
    net_transposed = net.transpose(2,1) # (batch_size, 1024, ..)
    batch_size = net_transposed.shape[0]
    num_proposal = net_transposed.shape[1]

    objectness_scores = net_transposed[:,:,0:2]
    end_points['objectness_scores'] = objectness_scores
    
    base_xyz = end_points['aggregated_vote_xyz'] # (batch_size, num_proposal, 3)
    center = base_xyz + net_transposed[:,:,2:5] # (batch_size, num_proposal, 3)
    end_points['center'] = center # with batch size
    return end_points

class VotePoseProposal(nn.Module):
    def __init__(self, num_proposal, sampling, seed_feat_dim=256):
        super().__init__() 

        self.num_proposal = num_proposal
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim

        # Vote clustering
        self.vote_aggregation = PointnetSAModuleVotes( 
                npoint=self.num_proposal,
                radius=0.3,
                nsample=16,
                mlp=[self.seed_feat_dim, 128, 128, 128],
                use_xyz=True,
                normalize_xyz=True
            )

        # Object proposal/detection
        # Objectness scores (2), center residual (3),
        # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)
        self.conv1 = torch.nn.Conv1d(128,128,1)
        self.conv2 = torch.nn.Conv1d(128,128,1)
        self.conv3 = torch.nn.Conv1d(128,2+3,1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(128)

    def forward(self, xyz, features, end_points):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """
        if self.sampling == 'vote_fps':
            # Farthest point sampling (FPS) on votes
            xyz, features, fps_inds = self.vote_aggregation(xyz, features)
            sample_inds = fps_inds
        elif self.sampling == 'seed_fps': 
            # FPS on seed and choose the votes corresponding to the seeds
            # This gets us a slightly better coverage of *object* votes than vote_fps (which tends to get more cluster votes)
            sample_inds = pointnet2_utils.furthest_point_sample(end_points['seed_xyz'], self.num_proposal)
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
        elif self.sampling == 'random':
            # Random sampling from the votes
            num_seed = end_points['seed_xyz'].shape[1]
            batch_size = end_points['seed_xyz'].shape[0]
            sample_inds = torch.randint(0, num_seed, (batch_size, self.num_proposal), dtype=torch.int).cuda()
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
        else:
            # log_string('Unknown sampling strategy: %s. Exiting!'%(self.sampling))
            exit()
        end_points['aggregated_vote_xyz'] = xyz # (batch_size, num_proposal, 3)
        end_points['aggregated_vote_inds'] = sample_inds # (batch_size, num_proposal,) # should be 0,1,2,...,num_proposal

        # --------- PROPOSAL GENERATION ---------
        net = F.relu(self.bn1(self.conv1(features)))
        
        net = F.relu(self.bn2(self.conv2(net))) 
        
        net = self.conv3(net) # (batch_size, 2+3, num_proposal)
        
        end_points = decode_scores(net, end_points)
        return end_points

def compute_vote_loss(end_points, gt_points):
    
    # seed_xyz = end_points['seed_xyz'] # if for seed loss, 
    #[B,N,3]
    vote_xyz = end_points['vote_xyz']
    dist1, ind1, dist2, ind2 = nn_distance(vote_xyz, gt_points) # 
    shift_loss = torch.mean(dist1)
    return shift_loss

def compute_proposal_loss(end_points, gt_points):
    center_proposal = end_points['center']
    objectness_proposal = end_points['objectness_scores']
    batch_size = center_proposal.shape[0]
    dist1, ind1, dist2, ind2 = nn_distance(center_proposal, gt_points)
    
    positive_mask = (dist1 <= 0.15*0.15)
    criterion = nn.CrossEntropyLoss()
    objectness_loss = criterion(objectness_proposal.reshape(-1,2), positive_mask.reshape(-1).long()) # 是否为关节点，交叉loss
    distance_loss = torch.mean(dist1[positive_mask])   #/ torch.sum(positive_mask) #* 100
    return objectness_loss, distance_loss

def get_loss(end_points, gt_points):
    # endpoints:[B,N,3]
    # gt_points:[B,M,3]

    # Vote loss
    vote_loss = compute_vote_loss(end_points, gt_points)
    end_points['vote_loss'] = vote_loss

    objectness_loss, distance_loss = compute_proposal_loss(end_points, gt_points)
    end_points['objectness_loss'] = objectness_loss
    end_points['distance_loss'] = distance_loss
    total_loss = vote_loss + objectness_loss + distance_loss
    end_points['total_loss'] = total_loss
    return total_loss, end_points