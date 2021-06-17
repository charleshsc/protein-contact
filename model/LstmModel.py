import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class BasicBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, dilation=1, residual=True, dropout_rate=0.2):
        super(BasicBlock, self).__init__()
        self.residual = residual
        self.in_channels = in_channels
        self.out_channels = out_channels

        # If in_channels != out_channels, then there is an additional cnn layer
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

        # CNN Layer
        self.conv = nn.Sequential(
            nn.Dropout(p=dropout_rate),
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
    

class LstmDilationModel(nn.Module):
    def __init__(self, hyper_params):
        super(LstmDilationModel, self).__init__()
        first_channels = hyper_params['residual_layers'][0][0]
        last_channels = hyper_params['residual_layers'][-1][1]

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=441,
            hidden_size=441,
            num_layers=1,
            bidirectional=True
        )

        # input CNN (Maxout Layer)
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=882,
                out_channels=first_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0)
            ),
            nn.InstanceNorm2d(num_features=first_channels),
            nn.ReLU(inplace=True)
        )

        # Middle dilation layers
        self.middle_layers = nn.ModuleList()
        for in_channels, out_channels, dilation, residual in hyper_params['residual_layers']:
            self.middle_layers.append(
                BasicBlock(in_channels, out_channels, dilation, residual, hyper_params['dropout_rate'])
            )

        # Final CNN (Output Layer)
        self.final_conv = nn.Conv2d(
                in_channels=last_channels,
                out_channels=10,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1)
            )

    def forward(self, x):
        """
            x: 1 x 441 x L x L
            out: 1 x 10 x L x L
        """

        # y: L x L x 441
        l = x.shape[2]
        y = x.squeeze(0)
        y = y.permute(2,1,0)

        output = []

        tot = y.shape[0] // 32
        if y.shape[0] > 300:
            for i in range(tot):
                left = i * 32
                right = (i + 1) * 32
                if i == tot - 1:
                    right = y.shape[0]
                inputs = y[left: right]
                output_r, (hn, cn) = self.lstm(inputs)
                output.append(output_r)
        else:
           for i in range(tot):
                left = i * 32
                right = (i + 1) * 32
                if i == tot - 1:
                    right = y.shape[0]
                inputs = y[left: right]
                output_r, (hn, cn) = self.lstm(inputs)
                output.append(output_r) 
        
        # y: L x L x 882
        y = torch.cat(output)
        y = y.permute(2,1,0)
        x = y.unsqueeze(0)

        # x: 1 x 882 x L x L
        middle = self.conv1_1(x)

        if x.shape[-1] > 380:
            for layer in self.middle_layers:
                middle = checkpoint(layer, middle)
        else:
            for layer in self.middle_layers:
                middle = layer(middle)

        middle = (middle + middle.transpose(2, 3)) / 2
        out = self.final_conv(middle)

        return out