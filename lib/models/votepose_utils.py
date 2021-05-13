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
# from utils.nn_distance import nn_distance

from models.pointnet2.pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule
from models.pointnet2 import pointnet2_utils

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
    num_proposal = net_transposed.shape[1] # predicted residue

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
                radius=0.3, # in meter
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

## panzhiyu revised version
def compute_vote_loss(end_points, gt_points):
    seed_xyz = end_points['seed_xyz'] # if for seed loss, 
    #[B,N,3]
    vote_xyz = end_points['vote_xyz']
    # import pdb; pdb.set_trace()
    batch_size = vote_xyz.shape[0]
    dist1, ind1, dist2, ind2 = nn_distance(seed_xyz, gt_points) # using the seed to judge its corresponding class
    positive_mask = (dist1 <= 0.49)
    # negative_mask = ~ positive_mask
    # ind1 is B * N  gt points is B * M * 3
    gt_vote_xyz = []
    for b in range(batch_size):
        gt_vote_xyz.append(gt_points[b:b+1,ind1[b],...])
    gt_vote_xyz = torch.cat(gt_vote_xyz,dim=0)
    # gt_vote_xyz = gt_points[]
    shift_distance = vote_xyz - gt_vote_xyz
    shift_loss = torch.mean(torch.norm(shift_distance,dim=2)[positive_mask])
    # if torch.sum(negative_mask) == 0:
    #     vote_loss = shift_loss
    # else:
    #     still_distance = vote_xyz - seed_xyz
    #     keep_loss = torch.mean(torch.norm(still_distance,dim=2)[negative_mask])
    #     vote_loss = keep_loss + shift_loss

    return shift_loss

def compute_proposal_loss(end_points, gt_points):
    center_proposal = end_points['center']
    objectness_proposal = end_points['objectness_scores']
    batch_size = center_proposal.shape[0]
    dist1, ind1, dist2, ind2 = nn_distance(center_proposal, gt_points)
    # import pdb; pdb.set_trace()
    positive_mask = (dist1 <= 0.15 * 0.15)  # for a big one at the initial 
    criterion = nn.CrossEntropyLoss()
    objectness_loss = criterion(objectness_proposal.reshape(-1,2), positive_mask.reshape(-1).long()) # 是否为关节点，交叉loss
    # if torch.sum(positive_mask) == 0:
    #     distance_loss = 0
    # else:
        # distance_loss = torch.mean(dist1[positive_mask]) #/ torch.sum(positive_mask) #* 100
    foreground_mask = (dist1 <= 0.49) # all positive points
    # import pdb;pdb.set_trace()
    distance_loss = torch.mean(dist1[foreground_mask])
    return objectness_loss, distance_loss

# def compute_vote_loss(end_points, gt_points):
#     vote_xyz = end_points['vote_xyz']
#     seed_xyz = end_points['seed_xyz']
#     #[B,N,3]
#     #vote_shift = vote_xyz - seed_xyz
#     batch_size = vote_xyz.shape[0]
#     shift_loss = 0
#     for b in range(batch_size):
#         #print(vote_xyz[b].shape)
#         #print(gt_points[b].shape)
#         dist1, ind1, dist2, ind2 = nn_distance(seed_xyz[b].unsqueeze(0), gt_points[b].unsqueeze(0)) # using seed's coord indx to supervise the vote
#         gt_seed_xyz = gt_points[b,ind1.squeeze(),:]
#         shift_loss_tmp = vote_xyz[b] - gt_seed_xyz #[B,N,3]
        
#         shift_loss += torch.sum(torch.norm(shift_loss_tmp, dim = 1)) / shift_loss_tmp.shape[0] 
#     #gt_points['vote_loss'] = shift_loss
#     return shift_loss

# def compute_proposal_loss(end_points, gt_points):
#     center_proposal = end_points['center']
#     objectness_proposal = end_points['objectness_scores']
#     batch_size = center_proposal.shape[0]
#     distance_loss = objectness_loss = 0
#     for b in range(batch_size):
#         dist1, ind1, dist2, ind2 = nn_distance(center_proposal[b].unsqueeze(0), gt_points[b].unsqueeze(0))
#         positive_mask = (dist1 <= 0.15).squeeze()
#         neg_mask = torch.zeros_like(positive_mask)
#         neg_mask[positive_mask==False] = 1
#         gt_mask = torch.stack([positive_mask,neg_mask],1)
#         criterion = nn.CrossEntropyLoss()
#         #print(gt_mask.float())
#         #print(objectness_proposal[b])
#         objectness_loss += criterion(objectness_proposal[b], positive_mask.long())
#         distance_loss += torch.sum(dist1[0,positive_mask]) / torch.sum(positive_mask.float()) * 10
#     return objectness_loss, distance_loss


def get_loss(end_points, gt_points):
    # endpoints:[B,N,3]
    # gt_points:[B,M,3]

    # Vote loss
    vote_loss = compute_vote_loss(end_points, gt_points)
    end_points['vote_loss'] = vote_loss

    objectness_loss, distance_loss = compute_proposal_loss(end_points, gt_points)
    # import pdb; pdb.set_trace()
    end_points['objectness_loss'] = objectness_loss
    end_points['distance_loss'] = distance_loss
    if distance_loss == 0:
        total_loss = vote_loss + objectness_loss
    else:
        total_loss = vote_loss + objectness_loss + distance_loss
    
    end_points['total_loss'] = total_loss
    return total_loss, end_points

## version 2
class VotePoseBackbone_(nn.Module):
    r"""
       Backbone network for point cloud feature learning.
       Based on Pointnet++ single-scale grouping network. 
        
       Parameters
       ----------
       input_feature_dim: int
            Number of input channels in the feature descriptor for each point.
            e.g. 3 for RGB.
    """
    def __init__(self, input_feature_dim=0):
        super().__init__()

        self.sa1 = PointnetSAModuleVotes(
                npoint=2048,
                radius=0.1,
                nsample=64,
                mlp=[input_feature_dim, 64, 64, 128],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa2 = PointnetSAModuleVotes(
                npoint=1024,
                radius=0.2,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa3 = PointnetSAModuleVotes(
                npoint=512,
                radius=0.3,
                nsample=16,
                mlp=[256, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa4 = PointnetSAModuleVotes(
                npoint=256,
                radius=0.4,
                nsample=16,
                mlp=[256, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.fp1 = PointnetFPModule(mlp=[256+256,256,256])
        self.fp2 = PointnetFPModule(mlp=[256+256,256,256])

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

        # --------- 4 SET ABSTRACTION LAYERS ---------
        xyz, features, fps_inds = self.sa1(xyz, features)
        end_points['sa1_inds'] = fps_inds
        end_points['sa1_xyz'] = xyz
        end_points['sa1_features'] = features

        xyz, features, fps_inds = self.sa2(xyz, features) # this fps_inds is just 0,1,...,1023
        end_points['sa2_inds'] = fps_inds
        end_points['sa2_xyz'] = xyz
        end_points['sa2_features'] = features

        xyz, features, fps_inds = self.sa3(xyz, features) # this fps_inds is just 0,1,...,511
        end_points['sa3_xyz'] = xyz
        end_points['sa3_features'] = features

        xyz, features, fps_inds = self.sa4(xyz, features) # this fps_inds is just 0,1,...,255
        end_points['sa4_xyz'] = xyz
        end_points['sa4_features'] = features

        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        features = self.fp1(end_points['sa3_xyz'], end_points['sa4_xyz'], end_points['sa3_features'], end_points['sa4_features'])
        features = self.fp2(end_points['sa2_xyz'], end_points['sa3_xyz'], end_points['sa2_features'], features)
        end_points['fp2_features'] = features
        end_points['fp2_xyz'] = end_points['sa2_xyz']
        num_seed = end_points['fp2_xyz'].shape[1]
        end_points['fp2_inds'] = end_points['sa1_inds'][:,0:num_seed] # indices among the entire input point clouds
        return end_points




class VotePoseVoting_(nn.Module):
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
        self.conv3 = torch.nn.Conv1d(self.in_dim, (3+self.out_dim) * self.vote_factor, 1) # vote 19 分支 # TODO: discription ability doublt
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
        # import pdb; pdb.set_trace()
        offset = net[:,:,:,0:3]
        vote_xyz = seed_xyz.unsqueeze(2) + offset # 同一个特征基础上直接归19类？
        #vote_xyz = vote_xyz.contiguous().view(batch_size, num_vote, 3)
        #vote_xyz = vote_xyz.transpose(2,1).contiguous()

        residual_features = net[:,:,:,3:] # (batch_size, num_seed, vote_factor, out_dim)
        vote_features = seed_features.transpose(2,1).unsqueeze(2) + residual_features # (B,seeds, factor, outdim)
        #vote_features = vote_features.contiguous().view(batch_size, num_vote, self.out_dim)
        vote_features = vote_features.transpose(1,3).contiguous()  # (B,outdim, factor,seeds)
        
        return vote_xyz, vote_features

class VotePoseProposal_(nn.Module):
    def __init__(self, num_proposal, sampling, vote_factor, seed_feat_dim=256):
        super().__init__() 

        self.num_proposal = num_proposal
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim
        self.vote_factor = vote_factor
        # Vote clustering
        self.vote_aggregation = PointnetSAModuleVotes( 
                npoint=self.num_proposal,
                radius=0.15,  # for 0.1
                nsample=128,
                mlp=[self.seed_feat_dim, 128, 128, 128],
                use_xyz=True,
                normalize_xyz=True
            )

        # Object proposal/detection
        # Objectness scores (2), center residual (3),
        # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)
        self.conv1 = torch.nn.Conv1d(128,128,1)
        self.conv2 = torch.nn.Conv1d(128,128,1)
        self.conv_op = torch.nn.ModuleList([torch.nn.Conv1d(128,2+3,1) for i in range(vote_factor)])
        #self.conv3 = torch.nn.Conv1d(128,2+3,1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(128)

    def forward(self, xyz, features, end_points):
        """
        Args:
            xyz: (B,M,19,3)
            features: (B,C,19,M)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """
        # Farthest point sampling (FPS) on votes
        xyz_ =[]
        features_ = []
        center_list = []
        for k in range(self.vote_factor): # divide in different joints and run the same network
            xyz_k, features_k, fps_inds = self.vote_aggregation(xyz[:,:,k,:].contiguous(), features[:,:,k,:].contiguous())
            net = F.relu(self.bn1(self.conv1(features_k)))
        
            net = F.relu(self.bn2(self.conv2(net))) 
        
            net = self.conv_op[k](net) # (batch_size, 2+3, num_proposal)
            objectness_scores, center = decode_scores_(net, xyz_k)
            xyz_.append(xyz_k)
            features_.append(features_k)
            center_list.append({'obj_score':objectness_scores, 'center':center})  # center batch * num_p * 3
                
        xyz_ = torch.stack(xyz_,dim = 1) #[batch, vote_factor, num_proposal, 3]
        features_ = torch.stack(features_,dim = 1) #[batch, vote_factor, feature_dim, 3]
        
        end_points['aggregated_vote_xyz'] = xyz_ # [batch, vote_factor, num_proposal, 3]
        end_points['center_list'] = center_list 
        return end_points

def decode_scores_(net, xyz):
    net_transposed = net.transpose(2,1) # (batch_size, 1024, ..)
    batch_size = net_transposed.shape[0]
    num_proposal = net_transposed.shape[1]

    objectness_scores = net_transposed[:,:,0:2]

    base_xyz = xyz 
    center = base_xyz + net_transposed[:,:,2:5] # (batch_size, num_proposal, 3)
    return objectness_scores, center

def compute_vote_loss_(end_points, gt_points):
    #gt_points:[B,num_joints,M,3]
    seed_xyz = end_points['seed_xyz'] # if for seed loss, 
    #[B,N,3]
    vote_xyz = end_points['vote_xyz']
    device = seed_xyz.device
    batch_size, num_joints = vote_xyz.shape[0], vote_xyz.shape[2]
    shift_loss = 0
    for k in range(num_joints):
        xyz_k = vote_xyz[:,:,k,:]
        gt_k = gt_points[:,:,k,:]
        dist1, ind1, dist2, ind2 = nn_distance(seed_xyz, gt_k) # using the seed to judge its corresponding class
        positive_mask = (dist1 <= 0.25 * 0.25).unsqueeze(2).repeat([1,1,3])  #0.25m内的vote点  # access illegal memory position
        
        gt_vote_k = []
        for b in range(batch_size):
            gt_vote_k.append(gt_k[b:b+1,ind1[b],...])
        gt_vote_k = torch.cat(gt_vote_k,dim=0)
        shift_distance = (xyz_k - gt_vote_k) * positive_mask
        
        shift_loss += torch.mean(torch.norm(shift_distance,dim=2))

    return shift_loss

def compute_proposal_loss_(end_points, gt_points):
    #gt_points:[B,num_joints,M,3]
    _, _, num_joints = gt_points.shape[0:3]  # 
    device = gt_points.device
    center_list = end_points['center_list']
    assert len(center_list) == num_joints
    criterion = nn.CrossEntropyLoss()
    distance_loss = torch.tensor(0).float().to(device)
    objectness_loss = torch.tensor(0).float().to(device)
    for k,center_dict in enumerate(center_list):
        gt_k = gt_points[:,:,k,:]
        center_proposal = center_dict['center']
        objectness_proposal = center_dict['obj_score']
        #print(center_proposal.shape)
        dist1, ind1, dist2, ind2 = nn_distance(center_proposal, gt_k)
        positive_mask = (dist1 <= 0.15 * 0.15)
        objectness_loss += criterion(objectness_proposal.reshape(-1,2), positive_mask.reshape(-1).long()) # Problem emerged 
        if torch.sum(positive_mask) != 0:
            # objectness_loss += criterion(objectness_proposal.reshape(-1,2), positive_mask.reshape(-1).long()) # Problem emerged  # assure there are multiple class input
            distance_loss += torch.mean(dist1[positive_mask])
        else:
            # vote_pose_mask = (dist1 <= 0.25*0.25)
            # objectness_loss += criterion(objectness_proposal.reshape(-1,2), positive_mask.reshape(-1).long()) # Problem emerged 
            distance_loss += torch.mean(dist1) # 
    return objectness_loss, distance_loss

def get_loss_(end_points, gt_points):
    # endpoints:[B,N,3]
    # gt_points:[B,M,3]

    # Vote loss
    vote_loss = compute_vote_loss_(end_points, gt_points)
    
    end_points['vote_loss'] = vote_loss

    objectness_loss, distance_loss = compute_proposal_loss_(end_points, gt_points)
    #import pdb; pdb.set_trace()
    end_points['objectness_loss'] = objectness_loss
    end_points['distance_loss'] = distance_loss
    # if ~distance_loss:
    #     total_loss = vote_loss + objectness_loss
    # else:

    if distance_loss == 0:
        total_loss = vote_loss + objectness_loss
    else:
        total_loss = vote_loss + objectness_loss + distance_loss
    
    end_points['total_loss'] = total_loss
    return total_loss, end_points






def huber_loss(error, delta=1.0):
    """
    Args:
        error: Torch tensor (d1,d2,...,dk)
    Returns:
        loss: Torch tensor (d1,d2,...,dk)

    x = error = pred - gt or dist(pred,gt)
    0.5 * |x|^2                 if |x|<=d
    0.5 * d^2 + d * (|x|-d)     if |x|>d
    Ref: https://github.com/charlesq34/frustum-pointnets/blob/master/models/model_util.py
    """
    abs_error = torch.abs(error)
    #quadratic = torch.min(abs_error, torch.FloatTensor([delta]))
    quadratic = torch.clamp(abs_error, max=delta)
    linear = (abs_error - quadratic)
    loss = 0.5 * quadratic**2 + delta * linear
    return loss

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
    
    if l1smooth:
        pc_dist = torch.sum(huber_loss(pc_diff, delta), dim=-1) # (B,N,M)
    elif l1:
        pc_dist = torch.sum(torch.abs(pc_diff), dim=-1) # (B,N,M)
    else:
        pc_dist = torch.sum(pc_diff**2, dim=-1) # (B,N,M)
    
    # import pdb; pdb.set_trace()
    dist1, idx1 = torch.min(pc_dist, dim=2) # (B,N)  每个推测点距离所有真值点最近的 距离 和真值indx
    dist2, idx2 = torch.min(pc_dist, dim=1) # (B,M)  每个真值点距离所有推测点最近的 推测Index 和 距离
    return dist1, idx1, dist2, idx2

def demo_nn_distance():
    np.random.seed(0)
    pc1arr = np.random.random((1,5,3))
    pc2arr = np.random.random((1,6,3))
    pc1 = torch.from_numpy(pc1arr.astype(np.float32))
    pc2 = torch.from_numpy(pc2arr.astype(np.float32))
    dist1, idx1, dist2, idx2 = nn_distance(pc1, pc2)
    print(dist1)
    print(idx1)
    dist = np.zeros((5,6))
    for i in range(5):
        for j in range(6):
            dist[i,j] = np.sum((pc1arr[0,i,:] - pc2arr[0,j,:]) ** 2)
    print(dist)
    print('-'*30)
    print('L1smooth dists:')
    dist1, idx1, dist2, idx2 = nn_distance(pc1, pc2, True)
    print(dist1)
    print(idx1)
    dist = np.zeros((5,6))
    for i in range(5):
        for j in range(6):
            error = np.abs(pc1arr[0,i,:] - pc2arr[0,j,:])
            quad = np.minimum(error, 1.0)
            linear = error - quad
            loss = 0.5*quad**2 + 1.0*linear
            dist[i,j] = np.sum(loss)
    print(dist)