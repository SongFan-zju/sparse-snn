"""Standard CIFAR ResNet34: 3x3 stem and BasicBlock depths [3,4,6,3]."""
from torch import nn

from src.models.resnet18_cifar import BasicBlock, ResNet18Cifar


class ResNet34Cifar(ResNet18Cifar):
    stage_depths = (3, 4, 6, 3)

    def __init__(self, num_classes=100, base_width=64, zero_init_residual=False):
        self._stage_index = 0
        super().__init__(num_classes, base_width, zero_init_residual)

    def _make_layer(self, channels, stride):
        depth = self.stage_depths[self._stage_index]
        self._stage_index += 1
        blocks = [BasicBlock(self.in_channels, channels, stride)]
        self.in_channels = channels
        blocks.extend(BasicBlock(channels, channels) for _ in range(depth - 1))
        return nn.Sequential(*blocks)


def resnet34_cifar_ann(num_classes=100, base_width=64, zero_init_residual=False):
    return ResNet34Cifar(num_classes, base_width, zero_init_residual)
