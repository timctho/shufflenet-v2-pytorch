from module import *
import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np


class ShuffleNetV2(nn.Module):
    """
    Class for building ShuffleNetV2 model with [0.5, 1.0, 1.5, 2.0] sizes
    """
    def __init__(self, height, width, in_channel, class_num, model_scale=1.0,
                 shuffle_group=2):
        super(ShuffleNetV2, self).__init__()

        self.class_num = class_num
        self.block_def = self._select_channel_size(model_scale)
        cur_channel = 24
        down_size = 4

        # First conv down size
        self.blocks = [('Init_Block',
                        nn.Sequential(
                            ConvBnRelu(in_channel, cur_channel, 3,
                                       stride=2, padding=1),
                            nn.MaxPool2d(3, stride=2, padding=1)
                        ))]

        # Middle shuffle blocks
        for idx, block in enumerate(self.block_def[:-1]):
            out_channel, repeat = block
            self.blocks += [('Stage{}_Block1'.format(idx + 2),
                             ShuffleNetV2Block(cur_channel, out_channel // 2,
                                               3, stride=2,
                                               shuffle_group=shuffle_group))]
            down_size *= 2
            for i in range(repeat - 1):
                self.blocks += [('Stage{}_Block{}'.format(idx + 2, i + 2),
                                 ShuffleNetV2Block(out_channel, out_channel,
                                                   3, shuffle_group=shuffle_group))]
            cur_channel = out_channel
        last_channel = self.block_def[-1][0]
        self.blocks += [('Conv', ConvBnRelu(cur_channel, last_channel, 1))]

        # Avg pool and predict
        pool_size = [np.ceil(height / down_size),
                     np.ceil(width / down_size)]
        self.blocks += [('AvgPool_Pred',
                         nn.Sequential(nn.AvgPool2d(pool_size, [1, 1]),
                                       nn.Conv2d(last_channel, class_num, 1)))]
        self.model = nn.Sequential(OrderedDict(self.blocks))

    def _select_channel_size(self, model_scale):
        # [(out_channel, repeat_times), (out_channel, repeat_times), ...]
        if model_scale == 0.5:
            return [(48, 4), (96, 8), (192, 4), (1024, 1)]
        elif model_scale == 1.0:
            return [(116, 4), (232, 8), (464, 4), (1024, 1)]
        elif model_scale == 1.5:
            return [(176, 4), (352, 8), (704, 4), (1024, 1)]
        elif model_scale == 2.0:
            return [(244, 4), (488, 8), (976, 4), (2048, 1)]
        else:
            raise ValueError('Unsupported model size.')

    def forward(self, x):
        out = self.model(x)
        out = out.view(-1, self.class_num)
        return out


class ShuffleResNetV2(nn.Module):
    """
    Class for building ShuffleNetV2-50 and SE-ShuffleNetV2-164
    """
    def __init__(self, height, width, in_channel, class_num, model_arch=50,
                 shuffle_group=2, use_se_block=True, se_reduction=16):
        super(ShuffleResNetV2, self).__init__()

        self.block_def = self._select_model_size(model_arch)
        self.class_num = class_num
        down_size = 2
        self.blocks = []

        # First conv down size
        self.init_block, cur_channel = self._get_init_block(model_arch, in_channel)
        self.blocks += self.init_block

        # Middle shuffle blocks
        for idx, block in enumerate(self.block_def[:-1]):
            out_channel, repeat = block
            down_size *= 2

            if idx == 0:
                self.blocks += [('Stage{}_Block1'.format(idx + 2),
                                 nn.MaxPool2d(3, stride=2, padding=1)),
                                ('Stage{}_Block2'.format(idx + 2),
                                 ShuffleNetV2ResBlock(cur_channel, out_channel // 2,
                                                      3, shuffle_group=shuffle_group,
                                                      use_se_block=use_se_block,
                                                      se_reduction=se_reduction)
                                 )]
                for i in range(repeat - 2):
                    self.blocks += [('Stage{}_Block{}'.format(idx + 2, i + 3),
                                     ShuffleNetV2ResBlock(out_channel, out_channel,
                                                          3, shuffle_group=shuffle_group,
                                                          use_se_block=use_se_block,
                                                          se_reduction=se_reduction
                                                          ))]
            else:
                self.blocks += [('Stage{}_Block1'.format(idx + 2),
                                 ShuffleNetV2ResBlock(cur_channel, out_channel // 2,
                                                      3, stride=2,
                                                      shuffle_group=shuffle_group,
                                                      use_se_block=use_se_block,
                                                      se_reduction=se_reduction
                                                      ))]
                for i in range(repeat - 1):
                    self.blocks += [('Stage{}_Block{}'.format(idx + 2, i + 2),
                                     ShuffleNetV2ResBlock(out_channel, out_channel,
                                                          3, shuffle_group=shuffle_group,
                                                          use_se_block=use_se_block,
                                                          se_reduction=se_reduction
                                                          ))]
            cur_channel = out_channel
        last_channel = self.block_def[-1][0]
        self.blocks += [('Conv', ConvBnRelu(cur_channel, last_channel, 1))]

        # Avg pool and predict
        pool_size = [np.ceil(height / down_size),
                     np.ceil(width / down_size)]
        self.blocks += [('AvgPool_Pred',
                         nn.Sequential(nn.AvgPool2d(pool_size, [1, 1]),
                                       nn.Conv2d(last_channel, class_num, 1)))]
        self.model = nn.Sequential(OrderedDict(self.blocks))

    def _get_init_block(self, model_arch, in_channel):
        out_channel = 64
        if model_arch == 50:
            blocks = [('Init_Block',
                       ConvBnRelu(in_channel, out_channel, 3,
                                  stride=2, padding=1)
                       )]
        elif model_arch == 164:
            blocks = [('Init_Block',
                       nn.Sequential(
                           ConvBnRelu(in_channel, out_channel, 3,
                                      stride=2, padding=1),
                           ConvBnRelu(out_channel, out_channel, 3,
                                      stride=1, padding=1),
                           ConvBnRelu(out_channel, 2 * out_channel, 3,
                                      stride=1, padding=1)
                       ))]
            out_channel *= 2
        else:
            raise ValueError('Support arch [50, 164]')
        return blocks, out_channel

    def _select_model_size(self, model_arch):
        # [(out_channel, repeat_times), (out_channel, repeat_times), ...]
        if model_arch == 50:
            return [(244, 4), (488, 4), (976, 6), (1952, 3), (2048, 1)]
        elif model_arch == 164:
            return [(340, 10), (680, 10), (1360, 23), (2720, 10), (2048, 1)]
        else:
            raise ValueError('Support arch [50, 164]')

    def forward(self, x):
        out = self.model(x)
        out = out.view(-1, self.class_num)
        return out

