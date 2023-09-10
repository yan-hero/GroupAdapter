import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log

# class ConvAdapter(nn.Module):
#     def __init__(self,mode='parallel'):
#
#         super(ConvAdapter, self).__init__()
#         self.mode = mode
#         self.convs = nn.ModuleList([nn.Conv1d(1,1,kernel_size=k,padding=int(k/2),bias=False) for k in range(3,10,2)])
#         self.kernel_sizes = [3,5,7,9]
#
#     def conv_block(self,k, x):
#         conv = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2), bias=False).cuda()
#         x_out = F.sigmoid(conv(x))
#         return x_out * x
#
#     def forward(self,x):
#
#         size = len(x.shape)
#         if size ==2:
#             x = x.unsqueeze(1) #[B,1,C]
#         identity = x
#         out = torch.zeros_like(x).to(x.device)
#
#         # if self.mode=='parallel':
#         #     conv_list = [F.sigmoid(conv(x)) for conv in self.convs]
#         #     for conv in conv_list:
#         #         out += conv*x
#         #     out += identity
#         #
#         # if self.mode=='sequential':
#         #     for conv in self.convs:
#         #         x_out = F.sigmoid(conv(x))
#         #         x = x_out*x
#         #     out = x+identity
#
#         if self.mode == 'parallel':
#             for k in self.kernel_sizes:
#                 out += self.conv_block(k,x)
#             out += identity
#
#         if self.mode=='sequential':
#             for k in self.kernel_sizes:
#                 x = self.conv_block(k,x)
#             out = x + identity
#
#         if size == 2:
#             out = out.squeeze(1)
#
#         return out




class Adapter(nn.Module):
    def __init__(self,in_size,hidden_size):
        super(Adapter, self).__init__()
        self.fc1 = nn.Linear(in_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,in_size)
        self.act = nn.ReLU()

    def forward(self,x):
        identity = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        return  x+identity


class GroupAdapter(nn.Module):
    def __init__(self,in_size,hidden_size,groups):
        
        super(GroupAdapter, self).__init__()
        self.fc_inter = nn.ModuleList([nn.Linear(int(in_size/groups),int(hidden_size/groups)).cuda() for _ in range(0,groups)])
        self.fc_outer = nn.ModuleList([nn.Linear(int(hidden_size/groups),int(in_size/groups)).cuda() for _ in range(0,groups)])
        self.act = nn.ModuleList([nn.ReLU() for _ in range(0,groups)])
        self.groups = groups

    def forward(self,x):
        identity = x
        C = x.shape[-1]
        out = torch.zeros_like(x)
        for i in range(0,self.groups):
            m = self.fc_inter[i](x[...,i*int(C/self.groups):(i+1)*int(C/self.groups)])
            m = self.act[i](m)
            m = self.fc_outer[i](m)
            out[...,i*int(C/self.groups):(i+1)*int(C/self.groups)] += m
        return out+identity








