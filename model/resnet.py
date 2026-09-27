import torch.nn as nn
from timm.layers import SelectAdaptivePool2d
from .module import BasicBlock, Bottleneck

class ResNet(nn.Module):
    def __init__(self, block, groups, num_classes=1000, pruned=False, pruned_cfg=None):
        super(ResNet, self).__init__()
        if not pruned:
            self.channels = 64
            self.block = block

            self.conv1 = nn.Conv2d(3, self.channels, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(self.channels)
            self.act1 = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

            self.layer1 = self._make_layer(channels=64, blocks=groups[0], strides=1, pruned=pruned, pruned_cfg=pruned_cfg["layer1"])
            self.layer2 = self._make_layer(channels=128, blocks=groups[1], strides=2, pruned=pruned, pruned_cfg=pruned_cfg["layer2"])
            self.layer3 = self._make_layer(channels=256, blocks=groups[2], strides=2, pruned=pruned, pruned_cfg=pruned_cfg["layer3"])
            self.layer4 = self._make_layer(channels=512, blocks=groups[3], strides=2, pruned=pruned, pruned_cfg=pruned_cfg["layer4"])

            self.global_pool = SelectAdaptivePool2d(pool_type='avg', flatten=True)

            num_features = 512 if self.block.message == "basic" else 512 * 4
            self.fc = nn.Linear(num_features, num_classes)
        else:
            assert pruned_cfg is not None
            self.channels = 64
            self.block = block
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=pruned_cfg["conv1"][0][0], kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(pruned_cfg["conv1"][0][0])
            self.act1 = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

            self.layer1 = self._make_layer(channels=64, blocks=groups[0], strides=1, pruned=pruned,
                                           pruned_cfg=pruned_cfg["layer1"])
            self.layer2 = self._make_layer(channels=128, blocks=groups[1], strides=2, pruned=pruned,
                                           pruned_cfg=pruned_cfg["layer2"])
            self.layer3 = self._make_layer(channels=256, blocks=groups[2], strides=2, pruned=pruned,
                                           pruned_cfg=pruned_cfg["layer3"])
            self.layer4 = self._make_layer(channels=512, blocks=groups[3], strides=2, pruned=pruned,
                                           pruned_cfg=pruned_cfg["layer4"])
            self.global_pool = SelectAdaptivePool2d(pool_type='avg', flatten=True)
            num_features = pruned_cfg["layer4"][-1][-1]
            self.fc = nn.Linear(num_features, num_classes)


    def _make_layer(self, channels, blocks, strides, pruned, pruned_cfg):
        list_strides = [strides] + [1] * (blocks-1)
        layer = nn.Sequential()
        for i in range(len(list_strides)):
            layer.add_module(str(i), self.block(in_channels=self.channels, out_channels=channels, strides=list_strides[i], pruned=pruned, pruned_cfg=pruned_cfg[i]))
            self.channels = channels if self.block.type == "basic" else channels * 4
        return layer

    def forward_features(self, x):
        x = self.maxpool(self.act1(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward_head(self, x):
        x = self.global_pool(x)
        x = self.fc(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def create_ResNet(depth, num_classes=1000, pruned=False, pruned_cfg=None):
    if depth == 18:
        return ResNet(BasicBlock, groups=[2, 2, 2, 2], num_classes=num_classes, pruned=pruned, pruned_cfg=pruned_cfg)
    elif depth == 50:
        return ResNet(Bottleneck, groups=[3, 4, 6, 3], num_classes=num_classes, pruned=pruned, pruned_cfg=pruned_cfg)
