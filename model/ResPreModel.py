import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
        BasicBlock for ResPre.
        Shape: 1 x C x L x L -> 1 x C x L x L
    """
    def __init__(self, channels=64):
        super(BasicBlock, self).__init__()

        # CNN Layer with IN
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1)
            ),
            nn.InstanceNorm2d(num_features=channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1)
            ),
            nn.InstanceNorm2d(num_features=channels)
        )

    def forward(self, x):
        """
            x: 1 x C x L x L
            out: 1 x C x L x L
        """
        y = self.conv(x)
        out = F.relu(y + x)

        return out
    

class ResPreModel(nn.Module):
    """
        ResPre Model.
        Shape: 1 x 441 x L x L -> 1 x 10 x L x L
    """
    def __init__(self):
        super(ResPreModel, self).__init__()

        # Input CNN
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

        # Middle Residual Layers
        self.middle_layers = nn.ModuleList()
        for _ in range(22):
            self.middle_layers.append(
                BasicBlock(channels=64)
            )

        # Final CNN -> Output
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
        for layer in self.middle_layers:
            middle = layer(middle)

        out = self.final_conv(middle)

        return out
