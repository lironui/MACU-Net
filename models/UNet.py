from torch import nn
import torch
from models.LAMNet import PositionLinearAttention, ChannelLinearAttention, LinearAttention


def conv3otherRelu(in_planes, out_planes, kernel_size=None, stride=None, padding=None):
    # 3x3 convolution with padding and relu
    if kernel_size is None:
        kernel_size = 3
    assert isinstance(kernel_size, (int, tuple)), 'kernel_size is not in (int, tuple)!'

    if stride is None:
        stride = 1
    assert isinstance(stride, (int, tuple)), 'stride is not in (int, tuple)!'

    if padding is None:
        padding = 1
    assert isinstance(padding, (int, tuple)), 'padding is not in (int, tuple)!'

    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
        nn.ReLU(inplace=True)  # inplace=True
    )


class UNetLAM(nn.Module):
    def __init__(self, band_num, class_num):
        super(UNetLAM, self).__init__()
        self.band_num = band_num
        self.class_num = class_num
        self.name = 'UNetLAM'

        channels = [32, 64, 128, 256, 512]
        self.conv1 = nn.Sequential(
            conv3otherRelu(self.band_num, channels[0]),
            conv3otherRelu(channels[0], channels[0]),
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            conv3otherRelu(channels[0], channels[1]),
            conv3otherRelu(channels[1], channels[1]),
        )

        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            conv3otherRelu(channels[1], channels[2]),
            conv3otherRelu(channels[2], channels[2]),
            conv3otherRelu(channels[2], channels[2]),
        )

        self.conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            conv3otherRelu(channels[2], channels[3]),
            conv3otherRelu(channels[3], channels[3]),
            conv3otherRelu(channels[3], channels[3]),
        )

        self.conv5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            conv3otherRelu(channels[3], channels[4]),
            conv3otherRelu(channels[4], channels[4]),
            conv3otherRelu(channels[4], channels[4]),
        )

        self.deconv4 = nn.ConvTranspose2d(channels[4], channels[3], kernel_size=(2, 2), stride=(2, 2))
        self.conv6 = nn.Sequential(
            conv3otherRelu(channels[4], channels[3]),
            conv3otherRelu(channels[3], channels[3]),
        )

        self.deconv3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=(2, 2), stride=(2, 2))
        self.conv7 = nn.Sequential(
            conv3otherRelu(channels[3], channels[2]),
            conv3otherRelu(channels[2], channels[2]),
        )

        self.deconv2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=(2, 2), stride=(2, 2))
        self.conv8 = nn.Sequential(
            conv3otherRelu(channels[2], channels[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv3otherRelu(channels[1], channels[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.deconv1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=(2, 2), stride=(2, 2))
        self.conv9 = nn.Sequential(
            conv3otherRelu(channels[1], channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv3otherRelu(channels[0], channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.lpa = PositionLinearAttention(channels[0])
        self.lca = ChannelLinearAttention()

        self.conv10 = nn.Conv2d(channels[0], self.class_num, kernel_size=1, stride=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        deconv4 = self.deconv4(conv5)
        conv6 = torch.cat((deconv4, conv4), 1)
        conv6 = self.conv6(conv6)

        deconv3 = self.deconv3(conv6)
        conv7 = torch.cat((deconv3, conv3), 1)
        conv7 = self.conv7(conv7)

        deconv2 = self.deconv2(conv7)
        conv8 = torch.cat((deconv2, conv2), 1)
        conv8 = self.conv8(conv8)

        deconv1 = self.deconv1(conv8)
        conv9 = torch.cat((deconv1, conv1), 1)
        conv9 = self.conv9(conv9)

        lpa = self.lpa(conv9)

        lca = self.lca(conv9)

        feat_sum = lpa + lca

        output = self.conv10(feat_sum)

        return output


class UNetLinear(nn.Module):
    def __init__(self, band_num, class_num):
        super(UNetLinear, self).__init__()
        self.band_num = band_num
        self.class_num = class_num
        self.name = 'UNetLinear'

        channels = [32, 64, 128, 256, 512]
        self.conv1 = nn.Sequential(
            conv3otherRelu(self.band_num, channels[0]),
            conv3otherRelu(channels[0], channels[0]),
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            conv3otherRelu(channels[0], channels[1]),
            conv3otherRelu(channels[1], channels[1]),
        )

        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            conv3otherRelu(channels[1], channels[2]),
            conv3otherRelu(channels[2], channels[2]),
            conv3otherRelu(channels[2], channels[2]),
        )

        self.conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            conv3otherRelu(channels[2], channels[3]),
            conv3otherRelu(channels[3], channels[3]),
            conv3otherRelu(channels[3], channels[3]),
        )

        self.conv5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            conv3otherRelu(channels[3], channels[4]),
            conv3otherRelu(channels[4], channels[4]),
            conv3otherRelu(channels[4], channels[4]),
        )

        self.deconv4 = nn.ConvTranspose2d(channels[4], channels[3], kernel_size=(2, 2), stride=(2, 2))
        self.conv6 = nn.Sequential(
            conv3otherRelu(channels[4], channels[3]),
            conv3otherRelu(channels[3], channels[3]),
        )

        self.deconv3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=(2, 2), stride=(2, 2))
        self.conv7 = nn.Sequential(
            conv3otherRelu(channels[3], channels[2]),
            conv3otherRelu(channels[2], channels[2]),
        )

        self.deconv2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=(2, 2), stride=(2, 2))
        self.conv8 = nn.Sequential(
            conv3otherRelu(channels[2], channels[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv3otherRelu(channels[1], channels[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.deconv1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=(2, 2), stride=(2, 2))
        self.conv9 = nn.Sequential(
            conv3otherRelu(channels[1], channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv3otherRelu(channels[0], channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.lpa = LinearAttention(channels[0])
        self.lca = ChannelLinearAttention()

        self.conv10 = nn.Conv2d(channels[0], self.class_num, kernel_size=1, stride=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        deconv4 = self.deconv4(conv5)
        conv6 = torch.cat((deconv4, conv4), 1)
        conv6 = self.conv6(conv6)

        deconv3 = self.deconv3(conv6)
        conv7 = torch.cat((deconv3, conv3), 1)
        conv7 = self.conv7(conv7)

        deconv2 = self.deconv2(conv7)
        conv8 = torch.cat((deconv2, conv2), 1)
        conv8 = self.conv8(conv8)

        deconv1 = self.deconv1(conv8)
        conv9 = torch.cat((deconv1, conv1), 1)
        conv9 = self.conv9(conv9)

        lpa = self.lpa(conv9)

        lca = self.lca(conv9)

        feat_sum = lpa + lca

        output = self.conv10(feat_sum)

        return output


class UNet(nn.Module):
    def __init__(self, band_num, class_num):
        super(UNet, self).__init__()
        self.band_num = band_num
        self.class_num = class_num
        self.name = 'UNet'

        channels = [32, 64, 128, 256, 512]
        self.conv1 = nn.Sequential(
            conv3otherRelu(self.band_num, channels[0]),
            conv3otherRelu(channels[0], channels[0]),
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            conv3otherRelu(channels[0], channels[1]),
            conv3otherRelu(channels[1], channels[1]),
        )

        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            conv3otherRelu(channels[1], channels[2]),
            conv3otherRelu(channels[2], channels[2]),
            conv3otherRelu(channels[2], channels[2]),
        )

        self.conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            conv3otherRelu(channels[2], channels[3]),
            conv3otherRelu(channels[3], channels[3]),
            conv3otherRelu(channels[3], channels[3]),
        )

        self.conv5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            conv3otherRelu(channels[3], channels[4]),
            conv3otherRelu(channels[4], channels[4]),
            conv3otherRelu(channels[4], channels[4]),
        )

        self.deconv4 = nn.ConvTranspose2d(channels[4], channels[3], kernel_size=(2, 2), stride=(2, 2))
        self.conv6 = nn.Sequential(
            conv3otherRelu(channels[4], channels[3]),
            conv3otherRelu(channels[3], channels[3]),
        )

        self.deconv3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=(2, 2), stride=(2, 2))
        self.conv7 = nn.Sequential(
            conv3otherRelu(channels[3], channels[2]),
            conv3otherRelu(channels[2], channels[2]),
        )

        self.deconv2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=(2, 2), stride=(2, 2))
        self.conv8 = nn.Sequential(
            conv3otherRelu(channels[2], channels[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv3otherRelu(channels[1], channels[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.deconv1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=(2, 2), stride=(2, 2))
        self.conv9 = nn.Sequential(
            conv3otherRelu(channels[1], channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv3otherRelu(channels[0], channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.conv10 = nn.Conv2d(channels[0], self.class_num, kernel_size=1, stride=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        deconv4 = self.deconv4(conv5)
        conv6 = torch.cat((deconv4, conv4), 1)
        conv6 = self.conv6(conv6)

        deconv3 = self.deconv3(conv6)
        conv7 = torch.cat((deconv3, conv3), 1)
        conv7 = self.conv7(conv7)

        deconv2 = self.deconv2(conv7)
        conv8 = torch.cat((deconv2, conv2), 1)
        conv8 = self.conv8(conv8)

        deconv1 = self.deconv1(conv8)
        conv9 = torch.cat((deconv1, conv1), 1)
        conv9 = self.conv9(conv9)

        output = self.conv10(conv9)

        return output


class UNetTiny(nn.Module):
    def __init__(self, band_num, class_num):
        super(UNetTiny, self).__init__()
        self.band_num = band_num
        self.class_num = class_num
        self.name = 'UNetTiny'

        channels = [32, 64, 128, 256]
        self.conv1 = nn.Sequential(
            conv3otherRelu(self.band_num, channels[0]),
            conv3otherRelu(channels[0], channels[0]),
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            conv3otherRelu(channels[0], channels[1]),
            conv3otherRelu(channels[1], channels[1]),
        )

        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            conv3otherRelu(channels[1], channels[2]),
            conv3otherRelu(channels[2], channels[2]),
            conv3otherRelu(channels[2], channels[2]),
        )

        self.conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            conv3otherRelu(channels[2], channels[3]),
            conv3otherRelu(channels[3], channels[3]),
            conv3otherRelu(channels[3], channels[3]),
        )

        self.deconv3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=(2, 2), stride=(2, 2))
        self.conv7 = nn.Sequential(
            conv3otherRelu(channels[3], channels[2]),
            conv3otherRelu(channels[2], channels[2]),
        )

        self.deconv2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=(2, 2), stride=(2, 2))
        self.conv8 = nn.Sequential(
            conv3otherRelu(channels[2], channels[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv3otherRelu(channels[1], channels[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.deconv1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=(2, 2), stride=(2, 2))
        self.conv9 = nn.Sequential(
            conv3otherRelu(channels[1], channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv3otherRelu(channels[0], channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.conv10 = nn.Conv2d(channels[0], self.class_num, kernel_size=1, stride=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        deconv3 = self.deconv3(conv4)
        conv7 = torch.cat((deconv3, conv3), 1)
        conv7 = self.conv7(conv7)

        deconv2 = self.deconv2(conv7)
        conv8 = torch.cat((deconv2, conv2), 1)
        conv8 = self.conv8(conv8)

        deconv1 = self.deconv1(conv8)
        conv9 = torch.cat((deconv1, conv1), 1)
        conv9 = self.conv9(conv9)

        output = self.conv10(conv9)

        return output
