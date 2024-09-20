from models.ConvMamba import ConvMamba
from models.pipeline import Classifier
from thop import profile
import json
import torch


with open('./params_use/convmamba.json', 'r') as fin:
    params = json.loads(fin.read())
dim = 64
depth = 4
patch_size = 15
mamba_param = params["mamba"]
d_state = mamba_param["d_state"]
d_conv = mamba_param["d_conv"]
expand = mamba_param["expand"]
params['data'] = params["data_PU"]
model = ConvMamba(params=params, d_model=dim, depth=depth, patch_size=patch_size, d_state=d_state, d_conv=d_conv, expand=expand).cuda()
input  = torch.randn(1, 225, 64).cuda()
flops, P = profile(model, inputs=(input, ))

# print('flops: ', flops / 1e9, 'G')
# print('params: ', P / 1e6, 'M')

pipeline = Classifier(params=params).cuda()
input  = torch.randn(1, 30, 15, 15).cuda()
flops, P = profile(pipeline, inputs=(input, ))

print('flops: ', flops / 1e9, 'G')
print('params: ', P / 1e6, 'M')