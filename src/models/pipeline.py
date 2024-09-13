import PIL
import time, json
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch.nn.init as init
from einops import rearrange, repeat
import collections
import torch.nn as nn
from models.ConvMamba import ConvMamba


def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight)


class Classifier(nn.Module):
    def __init__(self, params):
        super(Classifier, self).__init__()
        self.params = params
        net_params = params['net']
        data_params = params['data']

        self.model_type = net_params.get("model_type", 0)

        num_classes = data_params.get("num_classes", 16)
        patch_size = data_params.get("patch_size", 13)
        self.spectral_size = data_params.get("spectral_size", 200)

        depth = net_params.get("depth", 1)
        heads = net_params.get("heads", 8)
        mlp_dim = net_params.get("mlp_dim", 8)
        kernal = net_params.get('kernal', 3)
        padding = net_params.get('padding', 1)
        dropout = net_params.get("dropout", 0)
        conv2d_out = 64
        dim = net_params.get("dim", 64)
        self.d_model = dim
        dim_heads = dim
        mlp_head_dim = dim
        
        image_size = patch_size * patch_size

        self.pixel_patch_embedding = nn.Linear(conv2d_out, dim)

        # self.local_trans_pixel = Transformer(dim=dim, depth=depth, heads=heads, dim_heads=dim_heads, mlp_dim=mlp_dim, dropout=dropout)
        mamba_param = params["mamba"]
        d_state = mamba_param["d_state"]
        d_conv = mamba_param["d_conv"]
        expand = mamba_param["expand"]
        self.local_trans_pixel = ConvMamba(params=params, d_model=dim, depth=depth, patch_size=patch_size, d_state=d_state, d_conv=d_conv, expand=expand)
        self.new_image_size = image_size

        self.pixel_pos_embedding = nn.Parameter(torch.randn(self.new_image_size, dim))
        self.pixel_pos_embedding_relative = nn.Parameter(torch.randn(self.new_image_size, dim))
        self.pixel_pos_scale = nn.Parameter(torch.ones(1) * 0.01)
        # self.center_weight = nn.Parameter(torch.ones(depth, 1, 1) * 0.01)
        self.center_weight = nn.Parameter(torch.ones(depth, 1, 1)*0.001)


        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=self.spectral_size, out_channels=conv2d_out, kernel_size=(kernal, kernal), padding=(padding,padding)),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            # featuremap 
            # nn.Conv2d(in_channels=conv2d_out,out_channels=dim,kernel_size=3,padding=1),
            # nn.BatchNorm2d(dim),
            # nn.ReLU()
        )

        self.cls_token_pixel = nn.Parameter(torch.randn(1, 1, dim))
        self.to_latent_pixel = nn.Identity()

        self.mlp_head =nn.Linear(dim, num_classes)
        torch.nn.init.xavier_uniform_(self.mlp_head.weight)
        torch.nn.init.normal_(self.mlp_head.bias, std=1e-6)
        self.dropout = nn.Dropout(0.1)

        linear_dim = dim * 2
        self.classifier_mlp = nn.Sequential(
            nn.Linear(dim, linear_dim),
            nn.BatchNorm1d(linear_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(linear_dim, num_classes),
        )
    
    def centerlize(self, x):
        x = rearrange(x, 'b s h w-> b h w s')
        b, h, w, s = x.shape
        center_w = w // 2
        center_h = h // 2
        center_pixel = x[:,center_h, center_w, :]
        center_pixel = torch.unsqueeze(center_pixel, 1)
        center_pixel = torch.unsqueeze(center_pixel, 1)
        x_pixel = x +  center_pixel
        x_pixel = rearrange(x_pixel, 'b h w s-> b s h w')
        return x_pixel
        
    
    
    def encoder_block(self, x):
        '''
        x: (batch, s, w, h), s=spectral, w=weigth, h=height
        '''
        x_pixel = x 

        b, s, w, h = x_pixel.shape
        img = w * h

        x_pixel = self.conv2d_features(x_pixel)  # 将patch embed
        x_pixel = rearrange(x_pixel, 'b s w h-> b (w h) s') # (batch, w*h, s)
        x_pixel = self.dropout(x_pixel)
        x_pixel, x_center = self.local_trans_pixel(x_pixel)   #(batch, image_size, dim)
        reduce_x = torch.mean(x_pixel, dim=1)  # [batch_size, dim]
        return x_center, reduce_x

    def forward(self, x):
        '''
        x: (batch, s, w, h), s=spectral, w=weigth, h=height

        '''
        logit_x, _ = self.encoder_block(x)
        return  self.classifier_mlp(logit_x)