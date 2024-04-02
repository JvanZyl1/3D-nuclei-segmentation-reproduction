import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConvolution(nn.Module):

    def __init__(self, in_channels, out_channels_first, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels_first, kernel_size, stride, padding),
            nn.BatchNorm3d(out_channels_first),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels_first, out_channels, kernel_size, stride, padding),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x


class Conv(nn.Module):

    def __init__(self, in_channels, kernel_size, stride, padding, filters):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, filters, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv(x)
        return x


class MaxPooling(nn.Module):

    def __init__(self, kernel_size, stride):
        super().__init__()
        self.max_pool = nn.MaxPool3d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        x = self.max_pool(x)
        return x


class Deconvolution(nn.Module):

    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diff_depth = x2.size()[2] - x1.size()[2]
        diff_height = x2.size()[3] - x1.size()[3]
        diff_width = x2.size()[4] - x1.size()[4]

        # Padding should be split evenly between the sides
        x1 = F.pad(x1, [diff_width // 2, diff_width - diff_width // 2,
                        diff_height // 2, diff_height - diff_height // 2,
                        diff_depth // 2, diff_depth - diff_depth // 2])

        x = torch.cat([x2, x1], dim=1)
        return x


class NSN(nn.Module):  # this time with double convolutions to make the code cleaner

    def __init__(self, n_channels):
        super(NSN, self).__init__()
        self.n_channels = n_channels

        self.max_pool = MaxPooling(kernel_size=2, stride=2)
        self.deconv = Deconvolution()
        self.conv = Conv(in_channels=32, kernel_size=1, stride=1, padding=0, filters=1)

        self.double_conv_1 = DoubleConvolution(n_channels, 16, 32, 3, 1, 1)
        self.double_conv_2 = DoubleConvolution(32, 32, 64, 3, 1, 1)
        self.double_conv_3 = DoubleConvolution(64, 64, 128, 3, 1, 1)
        self.double_conv_4 = DoubleConvolution(192, 64, 64, 3, 1, 1)
        self.double_conv_5 = DoubleConvolution(96, 32, 32, 3, 1, 1)

    def forward(self, x):
        x1 = self.double_conv_1(x)
        x2 = self.max_pool(x1)
        x3 = self.double_conv_2(x2)
        x4 = self.max_pool(x3)
        x5 = self.double_conv_3(x4)
        x6 = self.deconv(x5, x4)
        x7 = self.double_conv_4(x6)
        x8 = self.deconv(x7, x1)
        x9 = self.double_conv_5(x8)
        x10 = self.conv(x9)

        return x10


class NDN(nn.Module):

    def __init__(self, n_channels):
        super(NDN, self).__init__()

        self.max_pool = MaxPooling(kernel_size=2, stride=2)
        self.deconv = Deconvolution()
        self.conv = Conv(in_channels=24, kernel_size=1, stride=1, padding=0, filters=1)

        self.double_conv_1 = DoubleConvolution(n_channels, 12, 24, 5, 1, 1)
        self.double_conv_2 = DoubleConvolution(24, 24, 48, 5, 1, 1)
        self.double_conv_3 = DoubleConvolution(48, 48, 96, 5, 1, 1)
        self.double_conv_4 = DoubleConvolution(96, 96, 192, 5, 1, 1)
        self.double_conv_5 = DoubleConvolution(192, 192, 384, 5, 1, 1)
        self.double_conv_6 = DoubleConvolution(576, 192, 192, 5, 1, 1)
        self.double_conv_7 = DoubleConvolution(288, 96, 96, 5, 1, 1)
        self.double_conv_8 = DoubleConvolution(144, 48, 48, 5, 1, 1)
        self.double_conv_9 = DoubleConvolution(72, 24, 24, 5, 1, 1)

    def forward(self, x):  # can't chain max pool and double conv because the output of double conv is needed
        x1 = self.double_conv_1(x)
        x2 = self.max_pool(x1)
        x3 = self.double_conv_2(x2)
        x4 = self.max_pool(x3)
        x5 = self.double_conv_3(x4)
        x6 = self.max_pool(x5)
        x7 = self.double_conv_4(x6)
        x8 = self.max_pool(x7)
        x9 = self.double_conv_5(x8)
        x10 = self.deconv(x9, x7)
        x11 = self.double_conv_6(x10)
        x12 = self.deconv(x11, x5)
        x13 = self.double_conv_7(x12)
        x14 = self.deconv(x13, x3)
        x15 = self.double_conv_8(x14)
        x16 = self.deconv(x15, x1)
        x17 = self.double_conv_9(x16)
        x18 = self.conv(x17)

        return x18
