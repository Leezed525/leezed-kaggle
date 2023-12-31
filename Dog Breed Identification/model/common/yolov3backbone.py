import torch.nn as nn
from .Conv import Conv
from .Conv import ResN


# just backbone train for 120 classes
# 100 epochs
# 0.95 train acc
# 0.03 val acc
class V3Backbone(nn.Module):
    def __init__(self):
        super(V3Backbone, self).__init__()
        # 416x416x3 -> 416x416x32 -> 208x208x64 -> 104x104x128 -> 52x52x256 -> 26x26x512 -> 13x13x1024
        self.dbl0_1 = Conv(3, 32, 3,  1,bias=True)
        self.dbl0_2 = Conv(32, 64, 3, 2, padding=1,bias=True)
        self.resn1_1 = ResN(64, 1)
        self.dbl1_1 = Conv(64, 128, 3, 2, padding=1,bias=True)
        self.resn2_2 = ResN(128, 2)
        self.dbl2_1 = Conv(128, 256, 3, 2, padding=1,bias=True)
        self.resn3_8 = ResN(256, 8)
        self.dbl3_1 = Conv(256, 512, 3, 2, padding=1,bias=True)
        self.resn4_8 = ResN(512, 8)
        self.dbl4_1 = Conv(512, 1024, 3, 2, padding=1,bias=True)
        self.resn5_4 = ResN(1024, 4)
        # self.out = nn.Linear(1024 * 13 * 13, 120)
        # nn.init.kaiming_normal_(self.out.weight)

    def forward(self, x):
        x = self.dbl0_1(x)
        x = self.dbl0_2(x)
        # print("before resn1_1: ", x.shape) 208x208x64
        x = self.resn1_1(x)
        x = self.dbl1_1(x)
        # print("before resn2_2: ", x.shape) 104x104x128
        x = self.resn2_2(x)
        x = self.dbl2_1(x)
        # print("before resn3_8: ", x.shape) 52x52x256
        x = self.resn3_8(x)
        x52 = x
        x = self.dbl3_1(x)
        # print("before resn4_8: ", x.shape) 26x26x512
        x = self.resn4_8(x)
        x26 = x
        x = self.dbl4_1(x)
        # print("before resn5_4: ", x.shape) 13x13x1024
        x = self.resn5_4(x)
        x13 = x
        # x = x.view(x.size(0), -1)
        # x = self.out(x)
        # print("x52: ", x52.shape)
        # print("x26: ", x26.shape)
        # print("x13: ", x13.shape)
        return x52, x26, x13
