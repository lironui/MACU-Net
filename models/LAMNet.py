import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.resnet as resnet
from torch.nn import Module, Conv2d, Parameter, Softmax


def l2_norm(x):
    return torch.einsum("bcn, bn->bcn", x, 1 / torch.norm(x, p=2, dim=-2))


def softplus_feature_map(x):
    return torch.nn.functional.softplus(x)


class LinearAttention(Module):
    def __init__(self, in_places, eps=1e-6):
        super(LinearAttention, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.in_places = in_places
        # self.exp_feature = exp_feature_map
        # self.tanh_feature = tanh_feature_map
        self.softplus_feature = softplus_feature_map
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

        Q = self.softplus_feature(Q).permute(-3, -1, -2)
        K = self.softplus_feature(K)

        # norm = 1 / torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps)
        # matrix = torch.einsum('bcn, bnm->bcm', V, Q)
        # matrix_sum = torch.einsum("bcm, bmn->bcn", matrix, K)
        #
        # weight_value = torch.einsum("bcn, bn->bcn", matrix_sum, norm)

        KV = torch.einsum("bcn, bmn->bcm", K, V)

        # att = torch.einsum("bnc, bcl->bnl", Q, K)
        norm = 1 / torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps)

        # weight_value = torch.einsum("bnm, bmc, bn->bcn", Q, KV, norm)
        weight_value = torch.einsum("bmc, bnm, bn->bcn", KV, Q, norm)
        weight_value = weight_value.view(batch_size, chnnels, height, width)

        return (x + self.gamma * weight_value).contiguous()


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


class LAMNet(nn.Module):
    def __init__(self, band, class_num, bn_momentum=0.01):
        super(LAMNet, self).__init__()
        self.name = 'LAMNet'
        self.Resnet101 = resnet.get_resnet101(band, dilation=[1, 1, 1, 2], bn_momentum=bn_momentum, is_fpn=False)
        self.Head = LAMNetHead(2048, class_num)

    def forward(self, x):
        size = x.size()[2:4]
        x = self.Resnet101(x)
        x = self.Head(x)
        output = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        return output


class LAMNetNone(nn.Module):
    def __init__(self, band, class_num, bn_momentum=0.01):
        super(LAMNetNone, self).__init__()
        self.name = 'LAMNetNone'
        self.Resnet101 = resnet.get_resnet101(band, dilation=[1, 1, 1, 2], bn_momentum=bn_momentum, is_fpn=False)
        self.Head = LAMNetHeadNone(2048, class_num)

    def forward(self, x):
        size = x.size()[2:4]
        x = self.Resnet101(x)
        x = self.Head(x)
        output = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        return output



class LAMNetHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LAMNetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())

        self.sa = PositionLinearAttention(inter_channels)
        self.sc = ChannelLinearAttention()
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)

        feat_sum = sa_conv + sc_conv

        sasc_output = self.conv8(feat_sum)

        return sasc_output


class LAMNetHeadNone(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LAMNetHeadNone, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_conv = self.conv51(feat1)

        feat2 = self.conv5c(x)
        sc_conv = self.conv52(feat2)

        feat_sum = sa_conv + sc_conv

        sasc_output = self.conv8(feat_sum)

        return sasc_output


if __name__ == '__main__':
    num_classes = 10
    in_batch, inchannel, in_h, in_w = 4, 3, 128, 128
    x = torch.randn(in_batch, inchannel, in_h, in_w)
    net = LAMNet50(3, class_num=num_classes)
    out = net(x)
    print(out.shape)