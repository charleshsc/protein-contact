"""
    PyTorch model. (FCN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, layers=2, dilation=1, add_res=True):
        super(ResidualBlock, self).__init__()
        assert kernel_size % 2 == 1

        self.add_res = add_res
        self.input_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=(1, 1)
        )

        self.conv_list = nn.ModuleList()
        for layer in range(layers):
            self.conv_list.append(
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=(kernel_size, kernel_size),
                    stride=(1, 1),
                    padding=(kernel_size//2*dilation, kernel_size//2*dilation),
                    dilation=(dilation, dilation)
                )
            )
            self.conv_list.append(
                nn.ReLU(inplace=True)
            )
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, inputs):
        x = self.input_conv(inputs)
        out = x
        for layer in self.conv_list:
            out = layer(out)

        if self.add_res:
            result = self.batch_norm(out + x)
        else:
            result = out
        return result


class ResNetModel(nn.Module):
    def __init__(self, hyper_params):
        super(ResNetModel, self).__init__()

        self.hyper_params = hyper_params
        self.middle_layers = hyper_params['residual_layers']
        assert self.middle_layers[0][0] == 64

        self.maxout_conv = nn.Conv2d(
            in_channels=441,
            out_channels=128,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0)
        )

        self.res_list = nn.ModuleList()
        for layer in self.middle_layers:
            assert len(layer) == 6
            self.res_list.append(
                ResidualBlock(
                    in_channels=layer[0],
                    out_channels=layer[1],
                    kernel_size=layer[2],
                    layers=layer[3],
                    dilation=layer[4],
                    add_res=layer[5]
                )
            )

        self.output_conv = nn.Conv2d(
            in_channels=self.middle_layers[-1][1],
            out_channels=10,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0)
        )

    def forward(self, inputs):
        # b x 441 x m x m
        m = inputs.shape[2]
        maxout: torch.Tensor = self.maxout_conv(inputs)

        # Element Wise Max Pooling
        maxout = maxout.reshape([-1, 64, 2, m, m])
        middle = torch.max(maxout, dim=2)[0]

        # Middle Layers
        for res_module in self.res_list:
            middle = res_module(middle)

        # Output Layer -> b x 10 x m x m
        out = self.output_conv(middle)

        return out


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
            assert len(layer) == 3 and layer[0] >= 1 and layer[0] % 2 == 1

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
            middle = torch.relu(middle_conv(middle))

        # Output Layer -> b x 10 x m x m
        out = self.output_conv(middle)

        return out
