import torch
import torch.nn as nn
import numpy


class AnimationTransformer(nn.Module):
    def __init__(self, in_nc=2, out_nc=256):
        super(AnimationTransformer, self).__init__()
        cnn = CNN(in_nc=in_nc, out_nc=out_nc)
        mlp = MLP()
        
    def forward(self, x):
        pass


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
    """
    A trivial 3-layer MLP, in superGlue in *ref, conv1d(stride=1) is used instead
    it is note worthy that conv1d and Linear might ensentially the same but could
    have some numerical difference(implicit), in this implementation, Linear is used
    ---
    *ref link: https://github.com/pallashadow/SuperGlue-pytorch/blob/master/models/superglue.py
    """
    def __init__(self, ):
        super(MLP, self).__init__()
        
        self.linear1 = nn.Sequential(nn.Linear(4, 64),
                                     nn.LeakyReLU(0.2, 1),
                                     nn.BatchNorm1d(64))
        self.linear2 = nn.Sequential(nn.Linear(64, 128),
                                     nn.LeakyReLU(0.2, 1),
                                     nn.BatchNorm1d(128))
        self.linear3 = nn.Sequential(nn.Linear(128, 256))

    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.linear2(x1)
        x3 = self.linear3(x2)
        return x3


def normalize_bbox(points): 
    pass



class Transformer(nn.Module):
    """
        Alternative Implementation of superGlue. Originals at 
        https://github.com/pallashadow/SuperGlue-pytorch/blob/master/models/superglue.py
    """
    def __init__(self, ):
        super(Transformer, self).__init__()
        self.multihead_attention = MultiheadAttention(d_model=256, head_num=80)
        
    
    def forward(self, x_ref, x_target):   # x_ref, x_target: torchsize([N, C, 256])
        self_attn = self.multihead_attention(q=x_ref, k=x_ref, v=x_ref)
        cross_attn = self.multihead_attention(q=x_ref, k=x_target, v=x_target)
        
        

class MultiheadAttention(nn.Module):
    def __init__(self, d_model=256, head_num=8):
        super(MultiheadAttention, self).__init__()
        self.head_num = head_num
        self.d_model = d_model
        self.dim = d_model // head_num
        
        self.attention = Attention()
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self.split(q), self.split(k), self.split(v)
        
        attn, _ = self.attention(q, k, v)     # torchsize([N, C, 8, 32])
        
        out = self.concat(attn) # torchsize([N, C, 256])
        out = self.w_concat(out)  
        
        return out
        
    def split(self, x):
        """
            x: torchsize([N, C, 256]) -> torchsize([N, C, 8, 32])
        """
        return x.view(x.shape[0], x.shape[1], self.head_num, self.dim)

    def concat(self, x):
        """
            x: torchsize([N, C, 8, 32]) -> torchsize([N, C, 256])
        """
        N, C, HEAD, DIM = x.shape
        return x.view(N, C, HEAD*DIM)


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, q, k, v):
        _, _, _, dim = k.shape   # [N, C, 8, 32]
        score = q @ k.transpose(2, 3) / numpy.sqrt(dim)
        
        score = self.softmax(score)
        v = score @ v
        
        return v, score
        