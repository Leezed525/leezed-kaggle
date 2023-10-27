import torch
import torch.nn as nn
from .BottleNeck import BottleNeckCSP
from .Conv import Conv


class YoloV3Predict(nn.Module):
    def __init__(self):
        super(YoloV3Predict, self).__init__()
        self.cv1 = BottleNeckCSP(128, 64, n=3, shortcut=True,bias=True)
        self.cv2 = Conv(64, 3 , 1, 1,bias=True)
        self.out = nn.Linear(3 * 52 * 52, 120)
        nn.init.kaiming_normal_(self.out.weight)


    def forward(self, x):
        x = self.cv1(x)
        x = self.cv2(x)
        x = x.view(x.size(0), -1)
        return x
