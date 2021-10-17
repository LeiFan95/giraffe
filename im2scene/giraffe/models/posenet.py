
import cv2
import numpy as np
from numpy.core.fromnumeric import var
import torch
from torch import nn
import torch.nn.functional as F
from im2scene.debugtool import var_shape
from im2scene.layers import Conv, Hourglass, Pool, Residual

class Merge(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Merge, self).__init__()
        self.conv = Conv(inp_dim, out_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)

class HeatmapLoss(torch.nn.Module):
    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, pred, gt):
        l = ((pred - gt)**2)
        l = l.mean(dim=3).mean(dim=2).mean(dim=1)
        return l

class PoseNet(nn.Module):
    def __init__(self, nstack, inp_dim, oup_dim, bn=False, increase=0, **kwargs):
        super(PoseNet, self).__init__()
        self.nstack = nstack
        self.oup_dim = oup_dim
        
        self.pre = nn.Sequential(
            Conv(3, 64, 7, 2, bn=True, relu=True),
            Residual(64, 128),
            Pool(2, 2),
            Residual(128, 128),
            Residual(128, inp_dim)
        )
        self.hgs = nn.ModuleList([
            nn.Sequential(
                Hourglass(4, inp_dim, bn, increase),
            ) for i in range(nstack)])
        self.features = nn.ModuleList([
            nn.Sequential(
                Residual(inp_dim, inp_dim),
                Conv(inp_dim, inp_dim, 1, bn=True, relu=True)
            ) for i in range(nstack)])
        self.outs = nn.ModuleList([Conv(inp_dim, oup_dim, 1, relu=False, bn=False) for i in range(nstack)])
        self.merge_features = nn.ModuleList([Merge(inp_dim, inp_dim) for i in range(nstack-1)])
        self.merge_preds = nn.ModuleList([Merge(oup_dim, inp_dim) for i in range(nstack-1)])
        self.nstack = nstack
        self.heatmapLoss = HeatmapLoss()

    def forward(self, imgs):
        # print('imgs', imgs)
        # var_shape('imgs', imgs)
        # input()
        # x = imgs.permute(0, 3, 1, 2) # x of size 1, 3, inp_dim, inp_dim
        x = imgs
        assert list(x.shape)[1] == 3, 'Input to posenet is incorrect!'

        x = self.pre(x)
        combined_hm_preds = []
        for i in range(self.nstack):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs[i](feature)
            combined_hm_preds.append(preds)
            if i < self.nstack - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)
        return torch.stack(combined_hm_preds, 1)

    def visualize(self, heatmaps, vis_size=64): # parts*size*size
        from im2scene.debugtool import draw_keypoints, var_shape
        # var_shape('heatmaps', heatmaps)
        # input()
        batch_size = heatmaps.shape[0]
        assert heatmaps.shape[2] == self.oup_dim, 'Not valid input for visulization'
        heatmaps_batch = heatmaps.detach().cpu().numpy()
        result = np.zeros((batch_size, vis_size, vis_size, 3))
        
        for idx, hm_cpu in enumerate(heatmaps_batch):
            keypoints = []
            hm_cpu = hm_cpu[-1]
            for heatmap in hm_cpu:
                heatmap = cv2.resize(heatmap, (vis_size, vis_size))
                ind = np.unravel_index(np.argmax(heatmap, axis=None), heatmap.shape)
                keypoints.append([ind[1], ind[0]])
            result[idx] = draw_keypoints(result[idx], keypoints)
        return result

    def calc_loss(self, combined_hm_preds, heatmaps):
        combined_loss = []
        for i in range(self.nstack):
            pred_resized = F.interpolate(combined_hm_preds[:, i], size=heatmaps.shape[-1])
            combined_loss.append(self.heatmapLoss(pred_resized, heatmaps))
        combined_loss = torch.stack(combined_loss, dim=1)
        return combined_loss