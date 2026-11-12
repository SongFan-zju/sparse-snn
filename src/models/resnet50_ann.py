from __future__ import annotations

import torch
from torch import nn


def _conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = _conv1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = _conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                _conv1x1(in_planes, planes * self.expansion, stride),
                nn.BatchNorm2d(planes * self.expansion),
            )
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x)
        return self.relu3(out)


class ResNet50Cifar(nn.Module):
    def __init__(self, num_classes: int = 10, base_width: int = 64, zero_init_residual: bool = True) -> None:
        super().__init__()
        self.in_planes = base_width
        self.conv1 = _conv3x3(3, base_width)
        self.bn1 = nn.BatchNorm2d(base_width)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(base_width, 3, stride=1)
        self.layer2 = self._make_layer(base_width * 2, 4, stride=2)
        self.layer3 = self._make_layer(base_width * 4, 6, stride=2)
        self.layer4 = self._make_layer(base_width * 8, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_width * 8 * Bottleneck.expansion, num_classes)
        self._init_weights()
        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, Bottleneck):
                    nn.init.zeros_(module.bn3.weight)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride_value in strides:
            layers.append(Bottleneck(self.in_planes, planes, stride_value))
            self.in_planes = planes * Bottleneck.expansion
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


def resnet50_cifar_ann(
    num_classes: int,
    base_width: int = 64,
    zero_init_residual: bool = True,
) -> ResNet50Cifar:
    return ResNet50Cifar(
        num_classes=num_classes,
        base_width=base_width,
        zero_init_residual=zero_init_residual,
    )
