import torch
import torch.nn as nn


class ShuffleUnit(nn.Module):
    def __init__(self, groups):
        super(ShuffleUnit, self).__init__()
        self.groups = groups

    def forward(self, x):
        n, c, h, w = x.shape
        x = x.reshape(n, self.groups, c // self.groups, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(n, c, h, w)
        return x


class ConvBnRelu(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1):
        super(ConvBnRelu, self).__init__()
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, dilation, groups,
                      False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True))

    def forward(self, x):
        return self.conv_bn_relu(x)


class ConvBn(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1):
        super(ConvBn, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, dilation, groups,
                      False),
            nn.BatchNorm2d(out_channel))

    def forward(self, x):
        return self.conv_bn(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, _, _ = x.size()
        y = self.avg_pool(x).view(n, c)
        y = self.fc(y).view(n, c, 1, 1)
        return x * y


class ShuffleNetV2Block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation=1, stride=1, shuffle_group=2):
        super(ShuffleNetV2Block, self).__init__()

        pad = (kernel_size // 2) * dilation
        self._stride = stride
        if stride == 1:
            # Split and concat unit
            if in_channel != out_channel:
                raise ValueError('in_c must equal out_c if stride is 1, which is {} and {}.'
                                 .format(in_channel, out_channel))
            branch_channel = (in_channel // 2) + (in_channel % 2)
            self._branch_channel = branch_channel
            self.branch = nn.Sequential(
                ConvBnRelu(branch_channel, branch_channel, 1),
                ConvBn(branch_channel, branch_channel, kernel_size, padding=pad, dilation=dilation,
                       groups=branch_channel),
                ConvBnRelu(branch_channel, branch_channel, 1)
            )
        else:
            # No split and downsample unit
            self.branch_0 = nn.Sequential(
                ConvBnRelu(in_channel, out_channel, 1),
                ConvBn(out_channel, out_channel, kernel_size, stride, padding=pad,
                       dilation=dilation, groups=out_channel),
                ConvBnRelu(out_channel, out_channel, 1)
            )
            self.branch_1 = nn.Sequential(
                ConvBn(in_channel, in_channel, kernel_size, stride, padding=pad, dilation=dilation,
                       groups=in_channel),
                ConvBnRelu(in_channel, out_channel, 1)
            )
        self.shuffle = ShuffleUnit(shuffle_group)

    def forward(self, x):
        if self._stride == 1:
            x_0, x_1 = torch.split(x, self._branch_channel, dim=1)
            out = torch.cat([self.branch(x_0), x_1], dim=1)
        else:
            out = torch.cat([self.branch_0(x), self.branch_1(x)], dim=1)
        out = self.shuffle(out)
        return out


class ShuffleNetV2ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation=1, stride=1,
                 shuffle_group=2, use_se_block=True, se_reduction=16):
        super(ShuffleNetV2ResBlock, self).__init__()

        pad = (kernel_size // 2) * dilation
        self._stride = stride
        self._in_channel = in_channel
        self._out_channel = out_channel
        if stride == 1 and in_channel == out_channel:
            # Split and concat unit
            branch_channel = (in_channel // 2) + (in_channel % 2)
            self._branch_channel = branch_channel
            self._blocks = [
                ConvBnRelu(branch_channel, branch_channel, 1),
                ConvBn(branch_channel, branch_channel, kernel_size, padding=pad, dilation=dilation,
                       groups=branch_channel),
                ConvBnRelu(branch_channel, branch_channel, 1)
            ]
            if use_se_block:
                self._blocks.append(SELayer(branch_channel, se_reduction))
            self.branch = nn.Sequential(*self._blocks)
        else:
            # No split and downsample unit
            self._blocks = [
                ConvBnRelu(in_channel, out_channel, 1),
                ConvBn(out_channel, out_channel, kernel_size, stride, padding=pad,
                       dilation=dilation, groups=out_channel),
                ConvBnRelu(out_channel, out_channel, 1)
            ]
            if use_se_block:
                self._blocks.append(SELayer(out_channel, se_reduction))
            self.branch_0 = nn.Sequential(*self._blocks)
            self.branch_1 = nn.Sequential(
                ConvBn(in_channel, in_channel, kernel_size, stride, padding=pad, dilation=dilation,
                       groups=in_channel),
                ConvBnRelu(in_channel, out_channel, 1)
            )
        self.shuffle = ShuffleUnit(shuffle_group)

    def forward(self, x):
        if self._stride == 1 and self._in_channel == self._out_channel:
            print(self._branch_channel, x.size())
            x_0, x_1 = torch.split(x, self._branch_channel, dim=1)
            x_0 = x_0 + self.branch(x_0)
            out = torch.cat([x_0, x_1], dim=1)
        else:
            out = torch.cat([self.branch_0(x), self.branch_1(x)], dim=1)
        out = self.shuffle(out)
        return out
