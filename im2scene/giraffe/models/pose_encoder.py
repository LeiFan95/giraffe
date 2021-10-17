import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from im2scene.debugtool import var_shape

class PoseEncoder(nn.Module):
    def __init__(self, input_ch=3, num_parts=17, hidden=256, 
                 stack=8, skips=[4]):
        super(PoseEncoder, self).__init__()
        self.input_ch = input_ch
        self.num_parts = num_parts
        self.input_pose_ch = self.num_parts*3
        self.hidden = hidden
        self.stack = stack
        self.skips = skips
        
        self.layers_in, self.layers_out = self.create_model()
    
    def create_model(self):
        layers = [nn.Linear(self.input_ch + self.input_pose_ch, self.hidden)]
        for i in range(self.stack - 1):
            layer = nn.Linear
            in_channels = self.hidden
            if i in self.skips:
                in_channels += self.input_ch
            layers += [layer(in_channels, self.hidden)]
        return nn.ModuleList(layers), nn.Linear(self.hidden, 3)
    
    def forward(self, x, keypoint_3d):
        keypoint_3d = keypoint_3d.unsqueeze(1).expand(-1, x.shape[1], -1)
        # var_shape('x', x)
        # var_shape('keypoint_3d', keypoint_3d)
        # input()
        h = torch.cat([x, keypoint_3d], dim=-1)
        for i, l in enumerate(self.layers_in):
            h = self.layers_in[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x, h], -1)
        return self.layers_out(h)