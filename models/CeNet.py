from torch import nn
import torch
import gc


class DualConvLayer(nn.Module):
    # 3x3x3 convolution with padding and relu
    def __init__(self, in_planes, out_planes):
        super(DualConvLayer, self).__init__()
        self.conv1a = nn.Sequential(
            nn.Conv3d(in_channels=in_planes, out_channels=out_planes, padding=0,
                      kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(inplace=True)
        )
        self.conv1b = nn.Sequential(
            nn.Conv3d(in_channels=out_planes, out_channels=out_planes, padding=1,
                      kernel_size=(3, 3, 3), stride=(1, 1, 1)),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(inplace=True)
        )
        self.conv2a = nn.Sequential(
            nn.Conv3d(in_channels=in_planes, out_channels=out_planes, padding=0,
                      kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(inplace=True)
        )
        self.conv2b = nn.Sequential(
            nn.Conv3d(in_channels=out_planes, out_channels=out_planes, padding=1,
                      kernel_size=(3, 3, 3), stride=(1, 1, 1)),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(inplace=True)
        )
        self.conv2c = nn.Sequential(
            nn.Conv3d(in_channels=out_planes, out_channels=out_planes, padding=1,
                      kernel_size=(3, 3, 3), stride=(1, 1, 1)),
            nn.BatchNorm3d(out_planes//4),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=out_planes, out_channels=out_planes, padding=0,
                      kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, X):
        x2l = self.conv1a(X)
        x2l = self.conv1b(x2l)
        x2r = self.conv2a(X)
        x2r = self.conv2b(x2r)
        x2r = self.conv2c(x2r)

        # print(x2l.shape, x2r.shape)
        x10 = torch.add(x2l, x2r)
        # print(x10.shape)
        return self.conv3(x10)


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1), bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1), bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2)),
            nn.Conv3d(ch_in, ch_out, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 下采样的gating signal 卷积
        g1 = self.W_g(g)
        del g
        # 上采样的 l 卷积
        x1 = self.W_x(x)
        # concat + relu
        psi = self.relu(g1 + x1)
        del g1, x1
        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi(psi)
        # 返回加权的 x
        return x * psi


class AttU_Net(nn.Module):
    def __init__(self, time_num=4, img_ch=4, output_ch=4):
        super(AttU_Net, self).__init__()
        self.name = 'AttU_Net'
        channels = [32, 64, 128, 256]

        self.Maxpool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.Conv1 = conv_block(img_ch, channels[0])
        self.Conv2 = conv_block(channels[0], channels[1])
        self.Conv3 = conv_block(channels[1], channels[2])
        self.Conv4 = conv_block(channels[2], channels[3])
        # self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        # self.Up5 = up_conv(ch_in=1024, ch_out=512)
        # self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        # self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=channels[3], ch_out=channels[2])
        self.Att4 = Attention_block(F_g=channels[2], F_l=channels[2], F_int=channels[1])
        self.Up_conv4 = conv_block(ch_in=channels[3], ch_out=channels[2])

        self.Up3 = up_conv(ch_in=channels[2], ch_out=channels[1])
        self.Att3 = Attention_block(F_g=channels[1], F_l=channels[1], F_int=channels[0])
        self.Up_conv3 = conv_block(ch_in=channels[2], ch_out=channels[1])

        self.Up2 = up_conv(ch_in=channels[1], ch_out=channels[0])
        self.Att2 = Attention_block(F_g=channels[0], F_l=channels[0], F_int=channels[0] // 2)
        self.Up_conv2 = conv_block(ch_in=channels[1], ch_out=channels[0])

        self.Conv_1x1 = nn.Conv3d(channels[0], output_ch, kernel_size=(time_num, 1, 1), stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
        del x

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        # x5 = self.Maxpool(x4)
        # x5 = self.Conv5(x5)
        #
        # # decoding + concat path
        # d5 = self.Up5(x5)
        # x4 = self.Att5(g=d5, x=x4)
        # d5 = torch.cat((x4, d5), dim=1)
        # d5 = self.Up_conv5(d5)
        # del x4, x5

        d4 = self.Up4(x4)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        del x3
        
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        del d4
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        del x2

        d2 = self.Up2(d3)
        del d3
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        del x1
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        del d2
        # d1 = self.sigmoid(d1)

        return d1.squeeze(-3)
