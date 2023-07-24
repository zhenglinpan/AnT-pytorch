import torch
import torch.nn as nn
import numpy as np


class AnimationTransformer(nn.Module):
    def __init__(self, embed_dim=256, head_num=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_num = head_num
        self.layer_names = ['self', 'cross'] * 2
        
        self.cnn = CNN(in_nc=2, out_nc=256)
        self.fcn = KeypointEncoder(embed_dim)
        self.gnn = AttentionalAggregation(embed_dim, head_num, self.layer_names)

    def forward(self, ref, target, ref_bboxes, target_bboxes, labels):
        """
            M is the number of segments founded
            :ref para: torchsize([M, 2, 32, 32])
            :target para: torchsize([M, 2, 32, 32])
            :ref_info: torchsize([M, 4])
            :target_info: torchsize([M, 4])
            :colors: torchsize([M, 1])
        """
        desc0 = self.cnn(ref) + self.fcn(ref_bboxes)  # torchsize([M, 256])
        desc1 = self.cnn(target) + self.fcn(target_bboxes)    # torchsize([M, 256])
        desc0, desc1 = self.gnn(desc0, desc1)   # torchsize([M, 256])
        
        sim_mat = desc0 @ desc1.transpose(1, 0) # torchsize([M, M])
        sim_mat = nn.Softmax(sim_mat, dim=-1)   # softmax for each row
        pred_color_ids = sim_mat @ labels # torchsize([M, 1])
        
        return pred_color_ids
        

class CNN(nn.Module):
    """
    A conventional CNN, as stated in the paper
    """
    def __init__(self, in_nc=2, out_nc=256):    # out_nc: D in the paper
        super(CNN, self).__init__()
        self.out_nc = out_nc
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_nc, 64, 3, 1, 1), 
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(0.2, 1))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), 
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(0.2, 1))
        self.conv3 = nn.Conv2d(64, 1, 3, 1, 1)
        
        self.conv_out = nn.Conv1d(out_nc * 4, out_nc, 1)
        
    def forward(self, x):   # x: torchsize([N, C, 32, 32]) in the paper
        N, C, H, W = x.shape
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2).view(N, C, H*W)     # reshape to torchsize([N, C, 1024])
        out = self.conv_out(x3)
        return out
        

class MLP(nn.Module):
    def __init__(self, feature_dims: list, bn=True):
        super(MLP, self).__init__()
        
        layer_num = len(feature_dims)
        model = []
        for i in range(layer_num - 1):
            model += [nn.Linear(feature_dims[i], feature_dims[i + 1])]
            if i < layer_num - 1:
                if bn: 
                    model += [nn.BatchNorm2d(feature_dims[i + 1])]
                model += [nn.LeakyReLU(0.2, 1)]
        
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


def normalize_bbox(points): 
    pass


class KeypointEncoder(nn.Module):
    """
        FCN in the paper
    """
    def __init__(self, embed_dim):
        super(KeypointEncoder, self).__init__()
        self.mlp = MLP([4, 32, 64, 128, embed_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)   # as per superGlue
        
    def forward(self, keypoints, color_ids):   # keypoints: torchsize([N, C, 4]), color_ids: torchsize([N, C, 1])
        return self.mlp(torch.cat([keypoints, color_ids], dim=2))


class AttentionalAggregation(nn.Module):
    def __init__(self, embed_dim, head_num, layer_names):
        super().__init__()
        self.names = layer_names
        self.trans_blocks = [AttentionalPropagation(embed_dim, head_num) for _ in range(len(layer_names))]
        
    def forward(self, desc_ref, desc_target): # desc_ref: torchsize([N, 256])
        for trans_block, name in zip(self.trans_blocks, self.names):
            if name == 'self':
                ref, target = desc_ref, desc_target
            elif name == 'cross':
                ref, target = desc_target, desc_ref
            
            desc_ref = desc_ref + trans_block(desc_ref, ref)
            desc_target = desc_target + trans_block(desc_target, target)
            
        return desc_ref, desc_target


class AttentionalPropagation(nn.Module):
    def __init__(self, embed_dim, head_num):
        super().__init__()
        self.multihead_attn = MultiheadAttention(embed_dim, head_num)
        self.mlp = MLP([embed_dim*2, embed_dim*2, embed_dim])
        
    def forward(self, ref, target): # ref: torchsize([N, 256])
        message = self.multihead_attn(q=ref, k=target, v=target)
        return self.mlp(torch.cat([ref, message], dim=2))


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, head_num=4):
        super(MultiheadAttention, self).__init__()
        self.head_num = head_num
        self.dim = embed_dim // head_num
        self.embed_dim = embed_dim
        
        self.attn = Attention()
        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.w_k = nn.Linear(embed_dim, embed_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)
        self.w_o = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, q, k, v): # x=q -> torchsize([N, 256]) as per paper
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self.split(q), self.split(k), self.split(v)
        
        v, _ = self.attn(q, k, v)
        
        o = self.merge(o)
        out = self.w_o(v)
        
        return out
    
    def split(self, x):
        N, C, _ = x.shape
        return x.view(N, C, self.head_num, self.dim)
        
    def merge(self, x):
        N, C, _, _ = x.shape
        return x.view(N, C, self.embed_dim)


class Attention(nn.Module):
    def __init__(self, ):
        super(Attention, self).__init__()
        
    def forward(self, q, k, v): # k: torchsize([N, C, 4, 64])
        attn = nn.Softmax(q @ k.transpose(2, 3) / np.sqrt(k.shape[3]), dim=-1)
        v = attn @ v
        
        return v, attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward).__init__()
        
        self.linear1 = nn.Linear(d_model, hidden)
        self.relu = nn.ReLU(1)
        self.dropout = nn.Dropout(p=drop_prob)
        self.linear2 = nn.Linear(hidden, d_model)
        
    def forward(self, x):
        x = self.dropout(self.relu(self.linear1(x)))
        x = self.linear2(x)
        return x