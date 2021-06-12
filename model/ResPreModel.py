import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, channels=64):
        super(BasicBlock, self).__init__()

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
        y = self.conv(x)
        out = F.relu(y + x)

        return out
    

class ResPreModel(nn.Module):
    def __init__(self):
        super(ResPreModel, self).__init__()

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

        self.middle_layers = nn.ModuleList()
        for _ in range(22):
            self.middle_layers.append(
                BasicBlock(channels=64)
            )

        self.final_conv = nn.Conv2d(
                in_channels=64,
                out_channels=10,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1)
            )

    def forward(self, x):
        middle = self.conv1_1(x)
        for layer in self.middle_layers:
            middle = layer(middle)

        out = self.final_conv(middle)

        return out
