import torch.nn as nn
import torch.nn.functional as F


class UpSampleBiLinear(nn.Module):
    def __init__(self, scale_factor=2):
        super(UpSampleBiLinear, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
