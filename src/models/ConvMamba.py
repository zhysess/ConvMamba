import torch
import torch.nn as nn
import math
from mamba_ssm import Mamba
import numpy as np
from einops import rearrange, repeat
import torch.nn.functional as F


class Mamba_encoder(nn.Module):
    def __init__(self,dim, d_state=16, d_conv=4, expand=2):
        super(Mamba_encoder, self).__init__()
        self.mamba = Mamba(  # This module uses roughly 3 * expand * d_model^2 parameters
                           d_model=dim,  # Model dimension d_model
                           d_state=d_state,  # SSM state expansion factor
                           d_conv=d_conv,  # Local convolution width
                           expand=expand,  # Block expansion factor
                           )

        self.norm = nn.LayerNorm(dim)
        self.act = nn.SiLU()
    def forward(self,x):
        
        residual = x
        x = self.mamba(x)
        x = self.norm(x)
        x = self.act(x)

        return residual + x


class ConvMamba(nn.Module):
    def __init__(self, params, d_model, depth, patch_size, d_state=16, d_conv=4, expand=2, ):
        super(ConvMamba, self).__init__()
        self.depth = depth
        self.params = params
        self.inner_conv1 = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1, stride=1)
        # self.inner_conv1 = nn.Conv3d(in_channels=d_model, out_channels=d_model, kernel_size=(3, 3, 3), padding=1, stride=1)
        self.conv = nn.Conv1d(in_channels=patch_size*patch_size, out_channels=1, kernel_size=1)
        self.layers = nn.ModuleList([
            Mamba_encoder(d_model, d_state=d_state, d_conv=d_conv, expand=expand) for _ in range(depth)
            ])
        self.dropout = nn.Dropout(0.1)
        self.fc_center = nn.Linear(d_model, d_model)
        self.fc_output = nn.Linear(d_model * 2, d_model)
    
    def forward(self, x):
        b, s, d = x.shape
        
        for i in range(self.depth):
            x_res = x
            x1 = x
            x1 = self.layers[i](x1)
            x1 = x1.reshape(b, int(np.sqrt(s)), int(np.sqrt(s)), d).permute(0, 3, 1, 2)
            x1 = F.relu(self.inner_conv1(x1))
            
            x1 = x1.permute(0, 2, 3, 1).reshape(b, s, d)
            x = x1 + x_res
            x = self.dropout(x)
        center_pixel = x[:, x.shape[1]//2, :]
        center_pixel = F.relu(self.fc_center(center_pixel))     
        output = torch.cat([self.conv(x).squeeze(dim=1), center_pixel], dim=1)
        output = self.fc_output(output)

        return x, output