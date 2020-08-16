from torch import nn
import torch
import torchsummary
from thop import profile
import torch.nn.functional as F
# from models.HANet import HANet_Conv
# from models.attention import PAM_Module, CAM_Module, CC_module


class StripPooling(nn.Module):
    """
    Reference:
    """
    def __init__(self, in_channels, pool_size, norm_layer=nn.BatchNorm2d, **up_kwargs):
        super(StripPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])
        self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))

        inter_channels = int(in_channels/4)
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_0 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                norm_layer(inter_channels))
        self.conv2_4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                norm_layer(inter_channels))
        self.conv2_5 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_6 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels*2, in_channels, 1, bias=False),
                                norm_layer(in_channels))
        # bilinear interpolate options
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x2_1 = self.conv2_0(x1)
        x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), (h, w), **self._up_kwargs)
        x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), (h, w), **self._up_kwargs)
        x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), (h, w), **self._up_kwargs)
        x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), (h, w), **self._up_kwargs)
        x1 = self.conv2_5(F.relu_(x2_1 + x2_2 + x2_3))
        x2 = self.conv2_6(F.relu_(x2_5 + x2_4))
        out = self.conv3(torch.cat([x1, x2], dim=1))
        return F.relu_(x + out)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


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


class DualConvLayer2D(nn.Module):
    # 3x3x3 convolution with padding and relu
    def __init__(self, in_planes, out_planes):
        super(DualConvLayer2D, self).__init__()
        self.conv1a = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=out_planes//2, padding=0,
                      kernel_size=1, stride=1),
            nn.BatchNorm2d(out_planes//2),
            nn.ReLU(inplace=True)
        )
        self.conv1b = nn.Sequential(
            nn.Conv2d(in_channels=out_planes//2, out_channels=out_planes//2, padding=1,
                      kernel_size=3, stride=1),
            nn.BatchNorm2d(out_planes//2),
            nn.ReLU(inplace=True)
        )
        self.conv2a = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=out_planes//2, padding=0,
                      kernel_size=1, stride=1),
            nn.BatchNorm2d(out_planes//2),
            nn.ReLU(inplace=True)
        )
        self.conv2b = nn.Sequential(
            nn.Conv2d(in_channels=out_planes//2, out_channels=out_planes//2, padding=1,
                      kernel_size=3, stride=1),
            nn.BatchNorm2d(out_planes//2),
            nn.ReLU(inplace=True)
        )
        self.conv2c = nn.Sequential(
            nn.Conv2d(in_channels=out_planes//2, out_channels=out_planes//2, padding=1,
                      kernel_size=3, stride=1),
            nn.BatchNorm2d(out_planes//2),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=out_planes, out_channels=out_planes, padding=0,
                      kernel_size=1, stride=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, X):
        x2l = self.conv1a(X)
        x2l = self.conv1b(x2l)
        x2r = self.conv2a(X)
        x2r = self.conv2b(x2r)
        x2r = self.conv2c(x2r)

        # print(x2l.shape, x2r.shape)
        x10 = torch.cat((x2l, x2r), dim=1)
        # print(x10.shape)
        return self.conv3(x10)


class TripleConvLayer2D(nn.Module):
    # 3x3x3 convolution with padding and relu
    def __init__(self, in_planes, out_planes):
        super(TripleConvLayer2D, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=out_planes, padding=0,
                      kernel_size=1, stride=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
        self.conv1a = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=out_planes, padding=0,
                      kernel_size=1, stride=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
        self.conv1b = nn.Sequential(
            nn.Conv2d(in_channels=out_planes, out_channels=out_planes, padding=1,
                      kernel_size=3, stride=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
        self.conv2a = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=out_planes, padding=0,
                      kernel_size=1, stride=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
        self.conv2b = nn.Sequential(
            nn.Conv2d(in_channels=out_planes, out_channels=out_planes, padding=1,
                      kernel_size=3, stride=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
        self.conv2c = nn.Sequential(
            nn.Conv2d(in_channels=out_planes, out_channels=out_planes, padding=1,
                      kernel_size=3, stride=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=out_planes*3, out_channels=out_planes, padding=0,
                      kernel_size=1, stride=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, X):
        x1 = self.conv0(X)
        x2l = self.conv1a(X)
        x2l = self.conv1b(x2l)
        x2r = self.conv2a(X)
        x2r = self.conv2b(x2r)
        x2r = self.conv2c(x2r)

        # print(x2l.shape, x2r.shape)
        x10 = torch.cat((x1, x2l, x2r), dim=1)
        # print(x10.shape)
        return self.conv3(x10)


class CMA2D(nn.Module):  # Channel Merge Attention
    def __init__(self, in_channels):
        super(CMA2D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1_1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1)  # // 4 考虑输出是否要除以4或6，当然如果除的话可以说是减少参数量
        self.relu = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)

    def forward(self, x):
        # global average pool
        x1 = self.avg_pool(x)
        x1 = self.conv1_1(x1)
        x1 = self.relu(x1)
        x1 = self.conv1_2(x1)
        x1 = self.sigmoid(x1)  # output N * C ?
        x2 = x * x1
        x2 = x + x2
        x2 = self.conv2(x2)  # 增加的
        return x2


class FGC2D(nn.Module):
    def __init__(self, band_num, class_num):
        super(FGC2D, self).__init__()
        self.band_num = band_num
        self.class_num = class_num
        self.name = 'FGC2D'

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
        )

        self.conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            conv3otherRelu(channels[2], channels[3]),
            conv3otherRelu(channels[3], channels[3]),
        )
        self.down_channel1 = nn.Conv2d(
            channels[3], channels[3], kernel_size=1, stride=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.down_channel2 = nn.Conv2d(
            channels[3], channels[3], kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.down_channel3 = nn.Conv2d(channels[3], channels[3], kernel_size=1, stride=1)

        self.deconv3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=(2, 2), stride=(2, 2))
        self.conv5 = nn.Sequential(
            CMA2D(channels[3]),
            conv3otherRelu(channels[3], channels[2]),
        )

        self.deconv2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=(2, 2), stride=(2, 2))
        self.conv6 = nn.Sequential(
            CMA2D(channels[2]),
            conv3otherRelu(channels[2], channels[1]),
        )

        self.deconv1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=(2, 2), stride=(2, 2))
        self.conv7 = nn.Sequential(
            CMA2D(channels[1]),
            conv3otherRelu(channels[1], channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.conv8 = nn.Conv2d(channels[0], self.class_num, kernel_size=1, stride=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        x1 = self.down_channel1(conv4)
        x2 = self.avg_pool(x1)
        x2 = self.down_channel2(x2)
        x2 = self.sigmoid(x2)  # output N * C ?
        x2 = x2 * x1 + x1
        x2 = self.down_channel3(x2)

        deconv3 = self.deconv3(x2)
        conv5 = torch.cat((deconv3, conv3), 1)
        conv5 = self.conv5(conv5)

        deconv2 = self.deconv2(conv5)
        conv6 = torch.cat((deconv2, conv2), 1)
        conv6 = self.conv6(conv6)

        deconv1 = self.deconv1(conv6)
        conv7 = torch.cat((deconv1, conv1), 1)
        conv7 = self.conv7(conv7)

        output = self.conv8(conv7)

        return output


class UNet2D(nn.Module):
    def __init__(self, band_num, class_num):
        super(UNet2D, self).__init__()
        self.band_num = band_num
        self.class_num = class_num
        self.name = 'UNet2D'

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


class MSFCN2D(nn.Module):  # 通道融合，全局池化
    def __init__(self, band_num, class_num):
        super(MSFCN2D, self).__init__()
        self.band_num = band_num
        self.class_num = class_num
        self.name = 'MSFCN2D'
        channels = [32, 64, 128, 256]

        self.conv1 = nn.Sequential(
            DualConvLayer2D(self.band_num, channels[0])
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            DualConvLayer2D(channels[0], channels[1])
        )

        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            DualConvLayer2D(channels[1], channels[2])
        )

        self.conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            DualConvLayer2D(channels[2], channels[3])
        )
        self.down_channel1 = nn.Conv2d(
            channels[3], channels[3], kernel_size=1, stride=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.down_channel2 = nn.Conv2d(
            channels[3], channels[3], kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.down_channel3 = nn.Conv2d(channels[3], channels[3], kernel_size=1, stride=1)

        self.deconv3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=(2, 2), stride=(2, 2))
        self.conv5 = nn.Sequential(
            CMA2D(channels[3]),
            conv3otherRelu(channels[3], channels[2]),
        )

        self.deconv2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=(2, 2), stride=(2, 2))
        self.conv6 = nn.Sequential(
            CMA2D(channels[2]),
            conv3otherRelu(channels[2], channels[1]),
        )

        self.deconv1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=(2, 2), stride=(2, 2))
        self.conv7 = nn.Sequential(
            CMA2D(channels[1]),
            conv3otherRelu(channels[1], channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.conv8 = nn.Conv2d(channels[0], self.class_num, kernel_size=1, stride=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        x1 = self.down_channel1(conv4)
        x2 = self.avg_pool(x1)
        x2 = self.down_channel2(x2)
        x2 = self.sigmoid(x2)
        x2 = x2 * x1 + x1
        x2 = self.down_channel3(x2)

        deconv3 = self.deconv3(x2)
        conv5 = torch.cat((deconv3, conv3), 1)
        conv5 = self.conv5(conv5)

        deconv2 = self.deconv2(conv5)
        conv6 = torch.cat((deconv2, conv2), 1)
        conv6 = self.conv6(conv6)

        deconv1 = self.deconv1(conv6)
        conv7 = torch.cat((deconv1, conv1), 1)
        conv7 = self.conv7(conv7)
        output = self.conv8(conv7)
        del conv1, conv2, conv3, conv4, conv5, conv6, conv7, deconv1, deconv2, deconv3, x, x1, x2

        return output


class DBFSFCN2D(nn.Module):  # 通道融合，全局池化
    def __init__(self, band_num, class_num):
        super(DBFSFCN2D, self).__init__()
        self.band_num = band_num
        self.class_num = class_num
        self.name = 'DBFSFCN2D'
        channels = [32, 64, 128, 256]

        self.conv1 = DualConvLayer2D(band_num, channels[0])

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            DualConvLayer2D(channels[0], channels[1])
        )

        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            DualConvLayer2D(channels[1], channels[2])
        )

        self.conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=( 2, 2)),
            DualConvLayer2D(channels[2], channels[3])
        )
        self.down_channel1 = nn.Conv2d(
            channels[3], channels[3], kernel_size=1, stride=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.down_channel2 = nn.Conv2d(
            channels[3], channels[3], kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.down_channel3 = nn.Conv2d(channels[3], channels[3], kernel_size=1, stride=1)

        self.deconv3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=(2, 2), stride=(2, 2))
        self.conv5 = nn.Sequential(
            CMA2D(channels[3]),
            conv3otherRelu(channels[3], channels[2]),
        )

        self.deconv2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=(2, 2), stride=(2, 2))
        self.conv6 = nn.Sequential(
            CMA2D(channels[2]),
            conv3otherRelu(channels[2], channels[1]),
        )

        self.deconv1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=(2, 2), stride=(2, 2))

        self.conv21 = DualConvLayer2D(band_num, channels[0])

        self.conv22 = DualConvLayer2D(channels[0], channels[0])

        self.conv23 = DualConvLayer2D(channels[0], channels[0])

        self.conv7 = nn.Sequential(
            CMA2D(channels[1]),
            conv3otherRelu(channels[1], channels[0]),
        )

        self.conv8 = nn.Sequential(
            conv3otherRelu(channels[1], channels[0], kernel_size=(3, 3), stride=(1, 1),
                           padding=(1, 1)),
        )

        self.conv9 = nn.Conv2d(channels[0], self.class_num, kernel_size=1, stride=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        x1 = self.down_channel1(conv4)
        x2 = self.avg_pool(x1)
        x2 = self.down_channel2(x2)
        x2 = self.sigmoid(x2)
        x2 = x2 * x1 + x1
        x2 = self.down_channel3(x2)

        deconv3 = self.deconv3(x2)
        conv5 = torch.cat((deconv3, conv3), 1)
        conv5 = self.conv5(conv5)

        deconv2 = self.deconv2(conv5)
        conv6 = torch.cat((deconv2, conv2), 1)
        conv6 = self.conv6(conv6)

        deconv1 = self.deconv1(conv6)
        conv7 = torch.cat((deconv1, conv1), 1)
        conv7 = self.conv7(conv7)

        conv21 = self.conv21(x)
        conv22 = self.conv22(conv21)
        conv23 = self.conv23(conv22)

        conv8 = torch.cat((conv7, conv23), 1)
        conv8 = self.conv8(conv8)
        output = self.conv9(conv8)

        return output


class DBTCFCN2D(nn.Module):  # 通道融合，全局池化
    def __init__(self, band_num, class_num):
        super(DBTCFCN2D, self).__init__()
        self.band_num = band_num
        self.class_num = class_num
        self.name = 'DBTCFCN2D'
        channels = [32, 64, 128, 256]

        self.conv1 = TripleConvLayer2D(band_num, channels[0])

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            TripleConvLayer2D(channels[0], channels[1])
        )

        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            TripleConvLayer2D(channels[1], channels[2])
        )

        self.conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=( 2, 2)),
            TripleConvLayer2D(channels[2], channels[3])
        )
        self.down_channel1 = nn.Conv2d(
            channels[3], channels[3], kernel_size=1, stride=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.down_channel2 = nn.Conv2d(
            channels[3], channels[3], kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.down_channel3 = nn.Conv2d(channels[3], channels[3], kernel_size=1, stride=1)

        self.deconv3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=(2, 2), stride=(2, 2))
        self.conv5 = nn.Sequential(
            CMA2D(channels[3]),
            conv3otherRelu(channels[3], channels[2]),
        )

        self.deconv2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=(2, 2), stride=(2, 2))
        self.conv6 = nn.Sequential(
            CMA2D(channels[2]),
            conv3otherRelu(channels[2], channels[1]),
        )

        self.deconv1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=(2, 2), stride=(2, 2))

        self.conv21 = TripleConvLayer2D(band_num, channels[0])

        self.conv22 = TripleConvLayer2D(channels[0], channels[0])

        self.conv23 = TripleConvLayer2D(channels[0], channels[0])

        self.conv7 = nn.Sequential(
            CMA2D(channels[1]),
            conv3otherRelu(channels[1], channels[0]),
        )

        self.conv8 = nn.Sequential(
            conv3otherRelu(channels[1], channels[0], kernel_size=(3, 3), stride=(1, 1),
                           padding=(1, 1)),
        )

        self.conv9 = nn.Conv2d(channels[0], self.class_num, kernel_size=1, stride=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        x1 = self.down_channel1(conv4)
        x2 = self.avg_pool(x1)
        x2 = self.down_channel2(x2)
        x2 = self.sigmoid(x2)
        x2 = x2 * x1 + x1
        x2 = self.down_channel3(x2)

        deconv3 = self.deconv3(x2)
        conv5 = torch.cat((deconv3, conv3), 1)
        conv5 = self.conv5(conv5)

        deconv2 = self.deconv2(conv5)
        conv6 = torch.cat((deconv2, conv2), 1)
        conv6 = self.conv6(conv6)

        deconv1 = self.deconv1(conv6)
        conv7 = torch.cat((deconv1, conv1), 1)
        conv7 = self.conv7(conv7)

        conv21 = self.conv21(x)
        conv22 = self.conv22(conv21)
        conv23 = self.conv23(conv22)

        conv8 = torch.cat((conv7, conv23), 1)
        conv8 = self.conv8(conv8)
        output = self.conv9(conv8)

        return output


class UNet2DPool(nn.Module):
    def __init__(self, band_num, class_num):
        super(UNet2DPool, self).__init__()
        self.band_num = band_num
        self.class_num = class_num
        self.name = 'UNet2DPool'

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

        self.pool = StripPooling(channels[4], (20, 12))

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

        conv5 = self.pool(conv5)

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


class UNet2DHA(nn.Module):
    def __init__(self, band_num, class_num):
        super(UNet2DHA, self).__init__()
        self.band_num = band_num
        self.class_num = class_num
        self.name = 'UNet2DHA'

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

        self.pool = StripPooling(channels[4], (20, 12))

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

        self.HANet1 = HANet_Conv(channels[0], channels[0])
        self.HANet2 = HANet_Conv(channels[1], channels[1])
        self.HANet3 = HANet_Conv(channels[2], channels[2])
        self.HANet4 = HANet_Conv(channels[3], channels[3])
        self.HANet5 = HANet_Conv(channels[4], channels[4])

    def forward(self, x):
        conv1 = self.conv1(x)
        conv1 = self.HANet1(conv1, conv1)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        # conv5 = self.HANet5(conv5, conv5)

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


class UNet2DAtt(nn.Module):
    def __init__(self, band_num, class_num):
        super(UNet2DAtt, self).__init__()
        self.band_num = band_num
        self.class_num = class_num
        self.name = 'UNet2DAtt'

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

        self.pool = StripPooling(channels[4], (20, 12))

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

        self.CCAtt = CC_module(channels[0])

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
        conv9 = self.CCAtt(conv9)

        output = self.conv10(conv9)

        return output


class UNet2MDAtt(nn.Module):
    def __init__(self, band_num, class_num):
        super(UNet2MDAtt, self).__init__()
        self.band_num = band_num
        self.class_num = class_num
        self.name = 'UNet2MDAtt'

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

        self.pool = StripPooling(channels[4], (20, 12))

        self.deconv4 = nn.ConvTranspose2d(channels[4], channels[3], kernel_size=(2, 2), stride=(2, 2))
        self.conv6 = nn.Sequential(
            CC_module(channels[4]),
            conv3otherRelu(channels[4], channels[3]),
            conv3otherRelu(channels[3], channels[3]),
        )

        self.deconv3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=(2, 2), stride=(2, 2))
        self.conv7 = nn.Sequential(
            CC_module(channels[3]),
            conv3otherRelu(channels[3], channels[2]),
            conv3otherRelu(channels[2], channels[2]),
        )

        self.deconv2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=(2, 2), stride=(2, 2))
        self.conv8 = nn.Sequential(
            CC_module(channels[2]),
            conv3otherRelu(channels[2], channels[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv3otherRelu(channels[1], channels[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.deconv1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=(2, 2), stride=(2, 2))
        self.conv9 = nn.Sequential(
            CC_module(channels[1]),
            conv3otherRelu(channels[1], channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv3otherRelu(channels[0], channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.conv10 = nn.Conv2d(channels[0], self.class_num, kernel_size=1, stride=1)

        self.CCAtt = CC_module(channels[0])

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
        conv9 = self.CCAtt(conv9)

        output = self.conv10(conv9)

        return output


if __name__ == '__main__':
    '''
    FCG2D,
    UNet2D
    '''
    time_num = 4
    band_num = 4
    class_num = 4
    model = UNet2D(time_num, band_num, class_num)
    flops, params = profile(model, input_size=(1, band_num*time_num, 256, 256))
    model.cuda()
    torchsummary.summary(model, (band_num*time_num, 256, 256))
    print('flops(G): %.3f' % (flops / 1e+9))
    print('params(M): %.3f' % (params / 1e+6))
