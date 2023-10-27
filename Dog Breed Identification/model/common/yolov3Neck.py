import torch.nn as nn
import torch
from .Conv import Conv
from .BottleNeck import BottleNeckCSP


class YoloV3Neck(nn.Module):
    def __init__(self):
        super(YoloV3Neck, self).__init__()
        # self.convolutionalset1 = ConvolutionalSet(1024, 512)
        self.convolutionalset1 = BottleNeckCSP(1024, 512, shortcut=True,n=5,bias=True)
        self.CV1 = BottleNeckCSP(512, 512, shortcut=True,bias=True)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.convolutionalset2 = BottleNeckCSP(1024, 256, shortcut=True,n=5,bias=True)
        self.CV2 = BottleNeckCSP(256, 256, shortcut=True,bias=True)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.convolutionalset3 = BottleNeckCSP(512, 128, shortcut=True,n=5,bias=True)
        self.CV3 = BottleNeckCSP(128, 128, shortcut=True,bias=True)
    def forward(self,x52,x26,x13):
        x = self.convolutionalset1(x13)
        x = self.CV1(x)
        x = self.up1(x)
        x = torch.cat((x, x26), dim=1)
        x = self.convolutionalset2(x)
        x = self.CV2(x)
        x = self.up2(x)
        x = torch.cat((x, x52), dim=1)
        x = self.convolutionalset3(x)
        x = self.CV3(x)
        # print("after neck : " , x.shape)  torch.Size([64, 128, 52, 52])
        return x



class ConvolutionalSet(nn.Module):
    def __init__(self, cin, cout):
        super(ConvolutionalSet, self).__init__()
        self.cv1 = Conv(cin, cout, 1, 1)
        self.cv2 = Conv(cout, cout * 2, 3, 1)
        self.cv3 = Conv(cout * 2, cout, 1, 1)
        self.cv4 = Conv(cout, cout * 2, 3, 1)
        self.cv5 = Conv(cout * 2, cout, 1, 1)

    def forward(self, x):
        return self.cv5(self.cv4(self.cv3(self.cv2(self.cv1(x)))))