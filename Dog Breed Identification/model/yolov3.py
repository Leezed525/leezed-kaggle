import torch.nn as nn
import torch
from .common.yolov3backbone import V3Backbone
from .common.yolov3Neck import YoloV3Neck
from .common.yolov3predict import YoloV3Predict



class YoloV3(nn.Module):
    def __init__(self):
        super(YoloV3, self).__init__()
        self.backbone = V3Backbone()
        self.neck = YoloV3Neck()
        self.predict = YoloV3Predict()
    def forward(self,x):
        x52, x26, x13 = self.backbone(x)
        x = self.neck(x52, x26, x13)
        x = self.predict(x)
        return x