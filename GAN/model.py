"""
    PyTorch model. (FCN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from utils import calc_pad

class Residual_Block(nn.Module):
    def __init__(self, channels, in_channels=None, dilation=None, kernel_size=3):
        super(Residual_Block, self).__init__()

        if dilation is None:
            # No dilation, use the padding to keep the size
            padding = calc_pad(kernel_size, 1)

            self.ops = nn.Sequential(
                nn.Conv2d(in_channels, channels, kernel_size=kernel_size, padding=padding),
                nn.PReLU(),
                nn.InstanceNorm2d(channels),
                nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, groups=channels),
                nn.PReLU(),
                nn.InstanceNorm2d(channels),
                nn.Dropout(0.25)
            )

        else:
            # with dilation, use the padding to keep the size
            padding = calc_pad(kernel_size, dilation)

            self.ops = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation,groups=channels),
                nn.PReLU(),
                nn.InstanceNorm2d(channels),
                nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation,groups=channels),
                nn.PReLU(),
                nn.InstanceNorm2d(channels),
                nn.Dropout(0.25)
            )

    def forward(self, x):
        '''
            x : bs x in_channel x width x height
            return : bs x out_channel x width x height
        '''
        residual = self.ops(x)
        return x + residual

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

# 从 b*441*L*L 到 b*10*L*L 产生结果
class Generator(nn.Module):
    def __init__(self, in_channels=441, out_channels=10, num_res_blocks = 2):
        super(Generator, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._num_res_blocks = num_res_blocks
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=9, padding=4),
            nn.PReLU()
        )
        model_sequence = nn.ModuleList()
        for i in range(num_res_blocks):
            model_sequence.append(Residual_Block(256,None,dilation=1,kernel_size=3))
            model_sequence.append(Residual_Block(256,None,dilation=2,kernel_size=3))
            model_sequence.append(Residual_Block(256,None,dilation=4,kernel_size=3))
        self.model = nn.Sequential(*model_sequence)

        self.blockSL = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1,groups=256),
            nn.PReLU()
        )

        self.blockL = nn.Sequential(
            nn.Conv2d(256, self._out_channels, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.fm = nn.Softmax(dim=1)
        self._initialize_weights()

    def forward(self, x):
        '''
            x: 1 x 441 x L x L -> 1 x 10 x L x L
        '''
        block1 = self.block1(x)
        res = self.model(block1)
        res = self.blockSL(res)
        blockL = self.blockL(block1 + res)
        res = self.fm(blockL)
        return res

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

# 理论上是输出每个位置的概率 feature和generator产生的结果结合 输出 每个位置的概率
class Discriminator(nn.Module):
    def __init__(self, in_channels=451, out_channels=10):
        super(Discriminator, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self.ops = nn.Sequential(
            nn.Conv2d(in_channels,256,kernel_size=3,padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1,groups=256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1,groups=256),
            nn.LeakyReLU()
        )
        self.finals = nn.Sequential(
            nn.Conv2d(256,out_channels,kernel_size=1,stride=1),
            nn.Sigmoid()
        )
        self._initialize_weights()

    def forward(self, feature, contact_map):
        '''
            feature: 1 x 441 x L x L
            contact_map: gt ( 1 x L x L ) or result (1 x 10 x L x L)
            return : 1 x
        '''
        if len(contact_map.shape) == 3:
            contact_map = F.one_hot(contact_map, num_classes=10).permute(0, 3, 1, 2).type(torch.float)
        x = torch.cat((feature,contact_map),dim=1)
        x = self.ops(x)
        x = self.finals(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
