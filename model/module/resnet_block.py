import torch.nn as nn

class BasicBlock(nn.Module):
    """
    basic building block for ResNet -18
    """
    type = "basic"
    def __init__(self, in_channels, out_channels, strides, pruned=False, pruned_cfg=None):
        super(BasicBlock, self).__init__()
        if not pruned:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=strides, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.act1 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.act2 = nn.ReLU(inplace=True)

            if strides is not 1:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides, padding=0, bias=False),
                    nn.BatchNorm2d(out_channels)
                )

        else:
            assert pruned_cfg is not None
            in_channels = pruned_cfg[0]
            mid_channels = pruned_cfg[1]
            out_channels = pruned_cfg[2]
            self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=strides, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(mid_channels)
            self.act1 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.act2 = nn.ReLU(inplace=True)
            self.downsample = None

            if strides is not 1:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides, padding=0, bias=False),
                    nn.BatchNorm2d(out_channels)
                )

    def forward(self, x):
        residual = x
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.downsample is not None:
            residual = self.downsample(residual)
        x = x + residual
        x = self.act2(x)
        return x


class Bottleneck(nn.Module):
    """
    Bottleneck block for ResNet-50, ResNet-101
    """
    type = "bottleneck"
    def __init__(self, in_channels, out_channels, strides, pruned=False, pruned_cfg=None):
        super(Bottleneck, self).__init__()
        if not pruned:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.act1 = nn.ReLU(inplace=True)

            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=strides, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.act2 = nn.ReLU(inplace=True)

            self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channels * 4)
            self.act3 = nn.ReLU(inplace=True)
            self.downsample = None

            if strides != 1 or in_channels != out_channels * 4:
                self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, 1, stride=strides, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        else:
            assert pruned_cfg is not None
            pruned_in_channels = pruned_cfg[0]
            pruned_mid_channels_1 = pruned_cfg[1]
            pruned_mid_channels_2 = pruned_cfg[2]
            pruned_out_channels = pruned_cfg[3]

            self.conv1 = nn.Conv2d(pruned_in_channels, pruned_mid_channels_1, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn1 = nn.BatchNorm2d(pruned_mid_channels_1)
            self.act1 = nn.ReLU(inplace=True)

            self.conv2 = nn.Conv2d(pruned_mid_channels_1, pruned_mid_channels_2, kernel_size=3, stride=strides, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(pruned_mid_channels_2)
            self.act2 = nn.ReLU(inplace=True)

            self.conv3 = nn.Conv2d(pruned_mid_channels_2, pruned_out_channels, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn3 = nn.BatchNorm2d(pruned_out_channels)
            self.act3 = nn.ReLU(inplace=True)

            self.downsample = None
            if strides != 1 or in_channels != out_channels * 4:
                self.downsample = nn.Sequential(
                    nn.Conv2d(pruned_in_channels, pruned_out_channels, kernel_size=1, stride=strides, padding=0, bias=False),
                    nn.BatchNorm2d(pruned_out_channels)
                )


    def forward(self, x):
        residual = x
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        if self.downsample is not None:
            residual = self.downsample(residual)
        x = x + residual
        x = self.act3(x)
        return x












