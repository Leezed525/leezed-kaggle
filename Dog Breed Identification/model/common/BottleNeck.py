import torch.nn as nn
import torch

from .Conv import Conv


def verify_computation(cin, cout, expansion):
    """
    Check if using bottleneck requires less computation than not using it (bottleneck is not always better) (comparing with Single 3*3 Conv)
    :param cin: (int) input channel
    :param cout: (int) output channel
    :param expansion: (float) expansion
    :return: True if using bottleneck requires less computation than not using it else False
    """
    c_ = int(cout * expansion)
    return 9 * c_ * c_ + c_ * (cin + cout) < 9 * cin * cout


class BottleNeck(nn.Module):
    def __init__(self, cin, cout, shortcut=False, expansion=0.5,bias=False):
        """
        BottleNect
        :param cin:
        :param cout:
        :param shortcut:
        :param expansion:
        """
        super(BottleNeck, self).__init__()
        c_ = int(cout * expansion)
        self.cv1 = Conv(cin, c_, 1, 1,bias=bias)
        self.cv2 = Conv(c_, cout, 3, 1,bias=bias)
        self.add = shortcut and cin == cout

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class BottleNeckCSP(nn.Module):
    def __init__(self, cin, cout, shortcut=True, expansion=0.5, n=1,bias=False):
        """
        BottleNeckCSP
        使用bottleNeck的CSP结构
        将输入分为两部分，一部分经过bottleNeck * n,另一部分经过Conv
        最后合并两部分经过一次Conv输出
        :param cin: (int) input channel
        :param cout: (int) output channel
        :param shortcut: (bool) shortcut
        :param expansion: (float) expansion
        :param n: (int) number of bottleNeck layers
        """
        super(BottleNeckCSP, self).__init__()
        c_ = int(cout * expansion)
        # 对于走多重bottleNeck的路，先经过一个Conv
        self.cv1 = Conv(cin, c_, 1, 1,bias=bias)
        # 在经过n个bottleNeck
        self.m = nn.Sequential(*[BottleNeck(c_, c_, shortcut, expansion,bias=bias) for _ in range(n)])
        # 在经过一个单纯的卷积层
        self.cv2 = nn.Conv2d(c_, c_, 1, 1)

        # 另一部分直接经过一个Conv
        self.cv3 = nn.Conv2d(cin, c_, 1, 1)

        # 最后合并两部分
        self.bn = nn.BatchNorm2d(2 * c_)
        self.activation = nn.SiLU()
        self.cv4 = Conv(2 * c_, cout, 1, 1)

    def forward(self, x):
        y1 = self.cv2(self.m(self.cv1(x)))
        y2 = self.cv3(x)
        return self.cv4(self.activation(self.bn(torch.cat((y1, y2), dim=1))))
