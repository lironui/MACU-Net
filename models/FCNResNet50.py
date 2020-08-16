import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.resnet as resnet
from torch.nn import Module, Conv2d, Parameter, Softmax


def l2_norm(x):
    return torch.einsum("bcn, bn->bcn", x, 1 / torch.norm(x, p=2, dim=-2))


class PositionLinearAttention(Module):
    def __init__(self, in_places, eps=1e-6):
        super(PositionLinearAttention, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.in_places = in_places
        self.l2_norm = l2_norm
        self.eps = eps

        self.query_conv = Conv2d(in_channels=in_places, out_channels=in_places // 8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_places, out_channels=in_places // 8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)

    def forward(self, x):
        # Apply the feature map to the queries and keys
        batch_size, chnnels, width, height = x.shape
        Q = self.query_conv(x).view(batch_size, -1, width * height)
        K = self.key_conv(x).view(batch_size, -1, width * height)
        V = self.value_conv(x).view(batch_size, -1, width * height)

        Q = self.l2_norm(Q).permute(-3, -1, -2)
        K = self.l2_norm(K)

        tailor_sum = 1 / (width * height + torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps))
        value_sum = torch.einsum("bcn->bc", V).unsqueeze(-1)
        value_sum = value_sum.expand(-1, chnnels, width * height)
        matrix = torch.einsum('bcn, bnm->bcm', V, Q)
        matrix_sum = value_sum + torch.einsum("bcm, bmn->bcn", matrix, K)

        weight_value = torch.einsum("bcn, bn->bcn", matrix_sum, tailor_sum)
        weight_value = weight_value.view(batch_size, chnnels, height, width)

        return (x + self.gamma * weight_value).contiguous()


class ChannelLinearAttention(Module):
    def __init__(self):
        super(ChannelLinearAttention, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        batch_size, chnnels, width, height = x.shape
        proj_query = x.view(batch_size, chnnels, -1)
        proj_key = x.view(batch_size, chnnels, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(batch_size, chnnels, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(batch_size, chnnels, height, width)

        out = self.gamma * out + x
        return out


class FCN50(nn.Module):
    def __init__(self, band, class_num, bn_momentum=0.01):
        super(FCN50, self).__init__()
        self.name = 'FCN50'
        self.Resnet50 = resnet.get_resnet50(band, dilation=[1, 1, 1, 2], bn_momentum=bn_momentum)
        self.Head = FCNHead(2048, class_num)

    def forward(self, x):
        size = x.size()[2:4]
        x = self.Resnet50(x)
        x = self.Head(x)
        output = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        return output


class FCNLAM50(nn.Module):
    def __init__(self, band, class_num, bn_momentum=0.01):
        super(FCNLAM50, self).__init__()
        self.name = 'FCNLAM50'
        self.Resnet50 = resnet.get_resnet50(band, dilation=[1, 1, 1, 2], bn_momentum=bn_momentum, is_fpn=False)
        self.Head = FCNHeadNone(2048, class_num)

    def forward(self, x):
        size = x.size()[2:4]
        x = self.Resnet50(x)
        x = self.Head(x)
        output = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        return output


class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FCNHead, self).__init__()
        inter_channels = in_channels // 4

        self.sa = PositionLinearAttention(inter_channels)
        self.sc = ChannelLinearAttention()

        self.conv5 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU())
        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        x = self.conv5(x)
        sa_feat = self.sa(x)
        sc_feat = self.sc(x)

        feat_sum = sa_feat + sc_feat

        sasc_output = self.conv6(feat_sum)

        return sasc_output


class FCNHeadNone(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FCNHeadNone, self).__init__()
        inter_channels = in_channels // 4
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU(),
                                   nn.Dropout2d(0.1, False),
                                   nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        return self.conv5(x)


if __name__ == '__main__':
    num_classes = 10
    in_batch, inchannel, in_h, in_w = 4, 3, 128, 128
    x = torch.randn(in_batch, inchannel, in_h, in_w)
    net = FCN50(3, class_num=num_classes)
    out = net(x)
    print(out.shape)