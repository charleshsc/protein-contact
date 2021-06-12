import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import padding
from torch.types import Device
from torch.utils.checkpoint import checkpoint
import numpy as np


class BasicBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, dilation=1, residual=True, dropout_rate=0.2):
        super(BasicBlock, self).__init__()
        self.residual = residual
        self.in_channels = in_channels
        self.out_channels = out_channels

        if self.in_channels != self.out_channels:
            self.input_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0)
                ),
                nn.InstanceNorm2d(num_features=out_channels)
            )

        self.conv = nn.Sequential(
            # nn.Dropout(p=dropout_rate),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(dilation, dilation),
                dilation=dilation
            ),
            nn.InstanceNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(dilation, dilation),
                dilation=dilation
            ),
            nn.InstanceNorm2d(num_features=out_channels)
        )

    def forward(self, x):
        if self.in_channels != self.out_channels:
            x = self.input_conv(x)

        y = self.conv(x)
        if self.residual:
            y = y + x
        out = F.relu(y)

        return out
    

class RegionalAttention(nn.Module):
    def __init__(self, region_size=3, dropout_rate=0.2, device='cuda'):
        super(RegionalAttention, self).__init__()
        self.conv_weight = torch.zeros([region_size, region_size, 1, 1, region_size*region_size])
        for i in range(region_size):
            for j in range(region_size):
                self.conv_weight[i,j,0,0,region_size*i+j] = 1
        self.conv_weight = self.conv_weight.permute(4, 3, 0, 1, 2).to(device)

        self.linear_q = nn.Linear(64, 64, bias=False)
        self.linear_k = nn.Linear(64, 64, bias=False)
        self.linear_v = nn.Linear(64, 64, bias=False)

        self.linear_o = nn.Linear(64, 64)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.region_size = region_size

    def reshape1(self, x):
        # 1 x L x L x 64 -> 4 x L x L x 16
        l = x.shape[1]
        x = x.reshape(l, l, 4, 16).permute(2, 0, 1, 3)
        return x

    def conv(self, x):
        # 4 x L x L x 16 -> 4 x L x L x 9 x 16
        x = x.unsqueeze(1)
        out = F.conv3d(
            input=x,
            weight=self.conv_weight,
            padding=(self.region_size//2, self.region_size//2, 0)
        )
        return out.permute(0, 2, 3, 1, 4)

    def forward(self, x: torch.Tensor):
        """
            x: 1 x 64 x L x L
        """
        # 4 x L x L x 16
        x = x.permute(0, 2, 3, 1)
        l = x.shape[1]
        s = x.shape[-1]
        q = self.reshape1(self.linear_q(x))
        k = self.reshape1(self.linear_k(x))
        v = self.reshape1(self.linear_v(x))
        
        # 4 x L x L x 9 x 16
        k = self.conv(k)
        v = self.conv(v)
        q = q.unsqueeze(3).expand_as(k)

        # 4 x L x L x 9
        qk = torch.sum(q * k, dim=-1) / np.sqrt(s)
        self.attention_score = torch.mean(qk, dim=0).detach()

        qk = self.dropout1(F.softmax(qk, dim=-1))

        # 4 x L x L x 9 x 16
        qk = qk.unsqueeze(-1).expand_as(v)

        # 1 x L x L x 64
        qkv = torch.sum(qk * v, dim=3).permute(1, 2, 0, 3).reshape(1, l, l, s)

        # 1 x 64 x L x L
        qkv = self.dropout2(self.linear_o(qkv)).permute(0, 3, 1, 2)

        return qkv


class AttentionModel(nn.Module):
    def __init__(self, hyper_params, return_score=False):
        super(AttentionModel, self).__init__()
        first_channels = hyper_params['residual_layers'][0][0]
        last_channels = hyper_params['residual_layers'][-1][1]
        self.return_score = return_score

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=441,
                out_channels=first_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0)
            ),
            nn.InstanceNorm2d(num_features=first_channels),
            nn.ReLU(inplace=True)
        )

        self.middle_layers = nn.ModuleList()
        for in_channels, out_channels, dilation, residual in hyper_params['residual_layers']:
            self.middle_layers.append(
                BasicBlock(in_channels, out_channels, dilation, residual, hyper_params['dropout_rate'])
            )

        self.final_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=last_channels,
                out_channels=64,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1)
            ),
            nn.InstanceNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        self.attention = RegionalAttention(
            dropout_rate=hyper_params['dropout_rate'],
            device=hyper_params['device'])

        self.final_conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=10,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )

    def forward(self, x):
        middle = self.conv1_1(x)

        if x.shape[-1] > 380:
            for layer in self.middle_layers:
                middle = checkpoint(layer, middle)
        else:
            for layer in self.middle_layers:
                middle = layer(middle)

        # Attention Model
        middle = self.final_conv1(middle)
        middle = self.attention(middle)
        out = self.final_conv2(middle)

        if not self.return_score:
            return out
        else:
            return out, self.attention.attention_score
