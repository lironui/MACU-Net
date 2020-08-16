from torch.utils.checkpoint import checkpoint

from .layers import *


class FCDenseNet(nn.Module):
    def __init__(self, in_channels=3, down_blocks=(5,5,5,5,5),
                 up_blocks=(5,5,5,5,5), bottleneck_layers=5,
                 growth_rate=16, out_chans_first_conv=48,
                 time_seires=4, n_classes=12, checkpoint=False, name='57'):
        super().__init__()
        self.name = 'tiramisu' + name
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        cur_channels_count = 0
        self.skip_connections = []
        skip_connection_channel_counts = []

        ## First Convolution ##

        self.add_module('firstconv', nn.Conv3d(in_channels=in_channels,
                  out_channels=out_chans_first_conv, kernel_size=(3, 3, 3),
                  stride=1, padding=(1, 1, 1), bias=True))
        cur_channels_count = out_chans_first_conv

        #####################
        # Downsampling path #
        #####################

        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        self.seBlocks = nn.ModuleList([])
        for i in range(len(down_blocks)):
            self.denseBlocksDown.append(
                DenseBlock(cur_channels_count, growth_rate, down_blocks[i]))
            cur_channels_count += (growth_rate*down_blocks[i])
            skip_connection_channel_counts.insert(0, cur_channels_count)
            self.transDownBlocks.append(TransitionDown(cur_channels_count))
            # self.seBlocks.append(CMA(cur_channels_count))


        #####################
        #     Bottleneck    #
        #####################

        self.add_module('bottleneck',Bottleneck(cur_channels_count,
                                     growth_rate, bottleneck_layers))
        prev_block_channels = growth_rate*bottleneck_layers
        cur_channels_count += prev_block_channels

        #######################
        #   Upsampling path   #
        #######################

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        for i in range(len(up_blocks)-1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]

            self.denseBlocksUp.append(DenseBlock(
                cur_channels_count, growth_rate, up_blocks[i], upsample=True))
            prev_block_channels = growth_rate*up_blocks[i]
            cur_channels_count += prev_block_channels

        ## Final DenseBlock ##

        self.transUpBlocks.append(TransitionUp(
            prev_block_channels, prev_block_channels))
        cur_channels_count = prev_block_channels + skip_connection_channel_counts[-1]

        self.denseBlocksUp.append(DenseBlock(
            cur_channels_count, growth_rate, up_blocks[-1],
                upsample=False))
        cur_channels_count += growth_rate*up_blocks[-1]

        ## Softmax ##

        self.finalConv = nn.Conv3d(in_channels=cur_channels_count,
               out_channels=n_classes, kernel_size=(time_seires, 1, 1), stride=1,
                   padding=0, bias=True)
        self.softmax = nn.LogSoftmax(dim=1)

    def stage0(self, x):
        return self.firstconv(x)

    def stage1(self, out):
        self.skip_connections = []
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            # skip_connections.append(self.seBlocks[i](out))
            self.skip_connections.append(out)
            out = self.transDownBlocks[i](out)
        return self.bottleneck(out)

    def stage2(self, out):
        return self.finalConv(out)

    def stage3(self, out):
        return self.finalConv(out)

    def forward(self, x):
        x = x + torch.zeros(1, dtype=x.dtype, device=x.device, requires_grad=True)
        if checkpoint:
            out = checkpoint(self.stage0, x)
            out = checkpoint(self.stage1, out)
            for i in range(len(self.up_blocks)):
                skip = self.skip_connections.pop()
                out = self.transUpBlocks[i](out, skip)
                del skip
                out = self.denseBlocksUp[i](out)

            out = checkpoint(self.stage2, out)
        else:
            out = self.stage0(x)
            out = self.stage1(out)
            out = self.stage2(out)

        # out = self.softmax(out)
        return out.squeeze(-3)


def FCDenseNet57(time_seires, band_num, n_classes, checkpoint=False):
    return FCDenseNet(
        in_channels=band_num, down_blocks=(2, 4, 8),
        up_blocks=(8, 4, 2), bottleneck_layers=4,
        growth_rate=16, out_chans_first_conv=32,
        time_seires=time_seires, n_classes=n_classes, checkpoint=checkpoint, name='57')


def FCDenseNet67(time_seires, band_num, n_classes, checkpoint=False):
    return FCDenseNet(
        in_channels=band_num, down_blocks=(5, 5, 5, 5, 5),
        up_blocks=(5, 5, 5, 5, 5), bottleneck_layers=5,
        growth_rate=16, out_chans_first_conv=48,
        time_seires=time_seires, n_classes=n_classes, checkpoint=checkpoint, name='67')


def FCDenseNet103(time_seires, band_num, n_classes, checkpoint=False):
    return FCDenseNet(
        in_channels=band_num, down_blocks=(4,5,7,10,12),
        up_blocks=(12,10,7,5,4), bottleneck_layers=15,
        growth_rate=16, out_chans_first_conv=48,
        time_seires=time_seires, n_classes=n_classes, checkpoint=checkpoint, name='103')
