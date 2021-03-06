import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class BasicBlock(nn.Module):
    """
        Basic dilation module.
        Shape: 1 x C1 x L x L -> 1 x C2 x L x L
    """
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
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(dilation, dilation),
                dilation=dilation
            ),
            nn.InstanceNorm2d(num_features=out_channels),
            nn.ELU(inplace=True),
            nn.Dropout(p=dropout_rate),
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
        """
            Input: 1 x C1 x L x L
            Output: 1 x C2 x L x L
        """
        if self.in_channels != self.out_channels:
            x = self.input_conv(x)

        y = self.conv(x)
        if self.residual:
            y = y + x
        out = F.elu(y)

        return out
    

class DeepModel(nn.Module):
    """
        Deep Dilation Model.
        Shape: 1 x 441 x L x L -> 1 x 10 x L x L
    """
    def __init__(self, hyper_params):
        super(DeepModel, self).__init__()

        # Input CNN Layer
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=441,
                out_channels=64,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0)
            ),
            nn.InstanceNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        # Middel Dilation Layers (60)
        self.middle_layers = nn.ModuleList()
        dilation = 1
        for i in range(60):
            self.middle_layers.append(
                BasicBlock(64, 64, dilation, True, hyper_params['dropout_rate'])
            )
            dilation *= 2
            if dilation == 32:
                dilation = 1

        # Final CNN Layer -> Output
        self.final_conv = nn.Conv2d(
                in_channels=64,
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
        middle = self.conv1_1(x)

        if x.shape[-1] > 330:
            for layer in self.middle_layers:
                middle = checkpoint(layer, middle)
        else:
            for layer in self.middle_layers:
                middle = layer(middle)

        out = self.final_conv(middle)

        return out
