from torch import nn
from .common.Conv import Conv
from .common.BottleNeck import BottleNeck


class FirstModel(nn.Module):
    def __init__(self, is_debug=False):
        super(FirstModel, self).__init__()
        self.is_debug = is_debug
        self.conv1 = Conv(3, 16, 3, 1)
        self.conv2 = Conv(16, 32, 3, 1)
        # 降低图像大小
        self.pool1 = nn.MaxPool2d(2, 2)
        # 通过卷积缩小图像大小
        self.conv3 = BottleNeck(32, 32, shortcut=True)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv4 = BottleNeck(32, 32, shortcut=True)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv5 = Conv(32, 16, 7, 5, 0)
        self.out = nn.Linear(16 * 6 * 6, 120)
        nn.init.kaiming_normal_(self.out.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.pool2(x)
        x = self.conv4(x)
        x = self.pool3(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x
