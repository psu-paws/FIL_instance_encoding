import torch
from torch import nn
import torchvision.transforms as transforms

# Code reimplementation of https://github.com/DmitryUlyanov/deep-image-prior/blob/master/models/skip.py
# Paper: https://openaccess.thecvf.com/content_cvpr_2018/Supplemental/2711-supp.pdf
class Hourglass(nn.Module):
    def __init__(self, in_c=32, num_channels_down=[16, 32, 64, 128, 128, 128],
                        num_channels_up=[128, 128, 128, 64, 32, 16],
                        num_channels_skip=[4, 4, 4, 4, 4, 4],
                        filter_size_down=[7, 7, 5, 5, 3, 3],
                        filter_size_up=[3, 3, 5, 5, 7, 7],
                        standarize=True,
                        sigmoid=False):
        super(Hourglass, self).__init__()
        
        self.standarize = standarize
        self.sigmoid = sigmoid
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.skips = nn.ModuleList()

        in_channel = in_c
        for out_channel, kernel_size in zip(num_channels_down, filter_size_down):
            down_block = []
            down_block.append(nn.Conv2d(in_channel, out_channel, kernel_size, stride=1, padding=(kernel_size - 1) // 2))
            down_block.append(nn.AvgPool2d(2, 2))
            down_block.append(nn.BatchNorm2d(out_channel))
            down_block.append(nn.LeakyReLU())

            down_block.append(nn.Conv2d(out_channel, out_channel, kernel_size, stride=1, padding=(kernel_size - 1) // 2))
            down_block.append(nn.BatchNorm2d(out_channel))
            down_block.append(nn.LeakyReLU(out_channel))
            self.downs.append(nn.Sequential(*down_block))
            in_channel = out_channel
        
        in_channel = in_c
        for i, out_channel in enumerate(num_channels_skip):
            skip_block = []
            skip_block.append(nn.Conv2d(in_channel, out_channel, 1))
            skip_block.append(nn.BatchNorm2d(out_channel))
            skip_block.append(nn.LeakyReLU())
            self.skips.append(nn.Sequential(*skip_block))
            in_channel = num_channels_down[i]
        
        for i, (out_channel, kernel_size) in enumerate(zip(num_channels_up, filter_size_up)):
            up_block = []
            if i == 0:
                in_channel = num_channels_skip[len(num_channels_skip) - 1 - i] + num_channels_down[-1]
            else:
                in_channel = num_channels_skip[len(num_channels_skip) - 1 - i] + num_channels_up[i - 1]
            up_block.append(nn.BatchNorm2d(in_channel))

            up_block.append(nn.Conv2d(in_channel, out_channel, kernel_size, padding=(kernel_size - 1) // 2))
            up_block.append(nn.BatchNorm2d(out_channel))
            up_block.append(nn.LeakyReLU())

            up_block.append(nn.Conv2d(out_channel, out_channel, 1))
            up_block.append(nn.BatchNorm2d(out_channel))
            up_block.append(nn.LeakyReLU())


            self.ups.append(nn.Sequential(*up_block))

        if sigmoid:
            self.out = nn.Sequential(nn.Conv2d(out_channel, 3, 1), nn.Sigmoid())
        else:
            self.out = nn.Conv2d(out_channel, 3, 1)

    def forward(self, x):
        out_downs = []
        out_skips = []
        #print(x.shape)
        for down_block, skip_block in zip(self.downs, self.skips):
            out_downs.append(down_block(x))
            out_skips.append(skip_block(x))
            x = out_downs[-1]
            #print(x.shape)
        for i, up_block in enumerate(self.ups):
            #print(nn.Upsample(scale_factor=2)(x).shape, out_skips[len(out_skips) - 1 - i].shape)
            #x = torch.cat([nn.Upsample(scale_factor=2)(x), out_skips[len(out_skips) - 1 - i]], dim=1)
            x = torch.cat([nn.Upsample(size=out_skips[len(out_skips) - 1 - i].shape[2:])(x), out_skips[len(out_skips) - 1 - i]], dim=1)
            #print(x.shape)
            x = up_block(x)
            #print(x.shape)
        #return self.out(x)
        out = self.out(x)
        # DIP works with standarized input with this term
        if self.standarize:
            transform = transforms.Normalize(torch.mean(out, dim=[2,3]).squeeze(), torch.std(out, dim=[2,3]).squeeze())
            return transform(out)
        else:
            return out


class InversionNet(nn.Module):
    def __init__(
        self,
        num_conv=3,
        upconv_channels=[(128, 'up'), (3, 'up')],
        in_c=128,
        last_activation='sigmoid'
    ):
        super(InversionNet, self).__init__()

        layers = []

        for _ in range(num_conv):
            layers.append(nn.Conv2d(in_c, in_c, 3, stride=1, padding=1))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        for i, (c, mode) in enumerate(upconv_channels):
            if mode == 'up':
                layers.append(nn.ConvTranspose2d(in_channels=in_c, out_channels=c, kernel_size=3, stride=2, padding=1, output_padding=1))
            elif mode == 'same':
                layers.append(nn.ConvTranspose2d(in_channels=in_c, out_channels=c, kernel_size=3, stride=1, padding=1, output_padding=0))
            if i != len(upconv_channels) - 1:
                layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            in_c = c

        # TODO: Try Tanh?
        if last_activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif last_activation == 'tanh':
            layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        #print(x.shape)
        for layer in self.layers:
            x = layer(x)
            #print(layer)
        return x
