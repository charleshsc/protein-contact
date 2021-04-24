"""
    PyTorch model. (FCN)
"""

import torch
import torch.nn as nn


class FCNModel(nn.Module):
    def __init__(self, hyper_params):
        super(FCNModel, self).__init__()

        self.hyper_params = hyper_params
        self.middle_layers = hyper_params['middle_layers']

        self.maxout_conv = nn.Conv2d(
            in_channels=441,
            out_channels=128,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0)
        )

        self.middle_conv_list = nn.ModuleList()

        for layer in self.middle_layers:
            assert layer >= 1 and layer % 2 == 1

            self.middle_conv_list.append(
                nn.Conv2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=(layer, layer),
                    stride=(1, 1),
                    padding=(layer//2, layer//2)
                )
            )

        self.output_conv = nn.Conv2d(
            in_channels=64,
            out_channels=10,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0)
        )

    def forward(self, inputs: torch.Tensor):
        # b x 441 x m x m
        m = inputs.shape[2]
        maxout: torch.Tensor = self.maxout_conv(inputs)

        # Element Wise Max Pooling
        maxout = maxout.reshape([-1, 64, 2, m, m])
        middle = torch.max(maxout, dim=2)[0]

        # Middle Layers
        for middle_conv in self.middle_conv_list:
            middle = middle_conv(middle)

        # Output Layer -> b x 10 x m x m
        out = self.output_conv(middle)

        return out
