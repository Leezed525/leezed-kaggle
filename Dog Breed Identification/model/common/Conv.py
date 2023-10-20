import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None, dilation=1):
    """
    Auto padding
    :param k:(int) kernel size
    :param p:(int) padding
    :param dilation:(int) dilation
    :return:
    """
    if dilation > 1:
        k = dilation * (k - 1) + 1
    if p is None:
        p = k // 2
    return p


class Conv(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding=None, dilation=1, groups=1, bias=False,
                 activation=nn.SiLU()):
        """
        Convolution 卷积层，不指定padding时自动padding(保持图像大小不变)
        :param cin: (int) input channel
        :param cout: (int) output channel
        :param kernel_size: (int) kernel size
        :param stride: (int) stride
        :param padding: (int) padding
        :param dilation: (int) dilation
        :param groups: (int) groups
        :param bias: (bool) bias
        :param activation: (nn.Module) activation function
        """
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size, stride, autopad(kernel_size, padding, dilation) if padding is None else padding, dilation,
                              groups, bias)
        self.bn = nn.BatchNorm2d(cout)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, cin, expansion=0.5):
        super(ResBlock, self).__init__()
        c_ = int(cin * expansion)
        self.cv1 = Conv(cin, c_, 1, 1)
        self.cv2 = Conv(c_, cin, 3, 1)

    def forward(self, x):
        return x + self.cv2(self.cv1(x))


class ResN(nn.Module):
    def __init__(self, cin, n=1):
        super(ResN, self).__init__()
        self.conv = Conv(cin, cin, 3, 1)
        self.m = nn.Sequential(*[ResBlock(cin) for _ in range(n)])

    def forward(self, x):
        return self.m(self.conv(x))
