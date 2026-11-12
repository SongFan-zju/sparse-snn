"""CIFAR-stem ResNet18 topology with independently addressable ReLU nodes.

The stem is a 3x3 stride-one convolution without a max pool. Four BasicBlock
stages have depths [2, 2, 2, 2] and widths [W, 2W, 4W, 8W]. The default
``base_width=64`` is the standard-width model; smaller widths are useful for
CPU conversion checks and are not separate pretrained sources.
"""
from __future__ import annotations

import torch
from torch import nn


def _conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = _conv3x3(in_channels, channels, stride)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = _conv3x3(channels, channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(channels),
            )
        self.relu2 = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.relu1(self.bn1(self.conv1(x)))
        residual = self.bn2(self.conv2(residual))
        return self.relu2(residual + self.shortcut(x))


class ResNet18Cifar(nn.Module):
    """Standard-width ResNet18 blocks with a CIFAR stem and configurable head."""

    def __init__(self, num_classes: int = 100, base_width: int = 64,
                 zero_init_residual: bool = False) -> None:
        super().__init__()
        if num_classes < 1 or base_width < 1:
            raise ValueError('num_classes and base_width must be positive')
        self.in_channels = base_width
        self.conv1 = _conv3x3(3, base_width)
        self.bn1 = nn.BatchNorm2d(base_width)
        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(base_width, stride=1)
        self.layer2 = self._make_layer(base_width * 2, stride=2)
        self.layer3 = self._make_layer(base_width * 4, stride=2)
        self.layer4 = self._make_layer(base_width * 8, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_width * 8, num_classes)
        self._init_weights()
        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, BasicBlock):
                    nn.init.zeros_(module.bn2.weight)

    def _make_layer(self, channels: int, stride: int) -> nn.Sequential:
        first = BasicBlock(self.in_channels, channels, stride)
        self.in_channels = channels
        return nn.Sequential(first, BasicBlock(channels, channels))

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0., std=.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        return self.fc(torch.flatten(out, 1))


def resnet18_cifar_ann(num_classes: int = 100, base_width: int = 64,
                       zero_init_residual: bool = False) -> ResNet18Cifar:
    return ResNet18Cifar(num_classes, base_width, zero_init_residual)
