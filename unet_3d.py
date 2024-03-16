import torch
import torch.nn as nn
import torch.nn.functional as F


class CBR(nn.Module):

    def __init__(self, in_channels, kernel_size, stride, padding, filters):
        super().__init__()
        self.cbr_block = nn.Sequential(
            nn.Conv3d(in_channels, filters, kernel_size, stride, padding),
            nn.BatchNorm3d(filters),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.cbr_block(x)
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


class DiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        smooth = 1e-6

        intersection = torch.sum(predictions * targets)

        dice_score = (2. * intersection + smooth) / (
                torch.sum(predictions) + torch.sum(targets) + smooth)

        return dice_score


class UNet3D(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.max_pool = MaxPooling(kernel_size=2, stride=2)
        self.deconv = Deconvolution()

        self.cbr_1 = CBR(in_channels=n_channels, kernel_size=3, stride=1, padding=1, filters=16)
        self.cbr_2 = CBR(in_channels=16, kernel_size=3, stride=1, padding=1, filters=32)
        self.cbr_3 = CBR(in_channels=32, kernel_size=3, stride=1, padding=1, filters=32)
        self.cbr_4 = CBR(in_channels=32, kernel_size=3, stride=1, padding=1, filters=64)
        self.cbr_5 = CBR(in_channels=64, kernel_size=3, stride=1, padding=1, filters=64)
        self.cbr_6 = CBR(in_channels=64, kernel_size=3, stride=1, padding=1, filters=128)
        self.cbr_7 = CBR(in_channels=192, kernel_size=3, stride=1, padding=1, filters=64)
        self.cbr_8 = CBR(in_channels=64, kernel_size=3, stride=1, padding=1, filters=64)
        self.cbr_9 = CBR(in_channels=96, kernel_size=3, stride=1, padding=1, filters=32)
        self.cbr_10 = CBR(in_channels=32, kernel_size=3, stride=1, padding=1, filters=32)
        self.conv = Conv(in_channels=32, kernel_size=1, stride=1, padding=0, filters=1)

    def forward(self, x):
        x1 = self.cbr_1(x)
        x2 = self.cbr_2(x1)
        x3 = self.max_pool(x2)
        x4 = self.cbr_3(x3)
        x5 = self.cbr_4(x4)
        x6 = self.max_pool(x5)
        x7 = self.cbr_5(x6)
        x8 = self.cbr_6(x7)
        x9 = self.deconv(x8, x5)
        x10 = self.cbr_7(x9)
        x11 = self.cbr_8(x10)
        x12 = self.deconv(x11, x2)
        x13 = self.cbr_9(x12)
        x14 = self.cbr_10(x13)
        x15 = self.conv(x14)
        probabilities = torch.sigmoid(x15)
        return probabilities
