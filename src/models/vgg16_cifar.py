"""QCFS author's CIFAR VGG16 topology, naming, and initialization.

The 13 biased Conv/BN/ReLU blocks and two hidden FC activations keep the
author's layer1..5/classifier indices. Installing scalar QCFS activations
therefore permits strict loading of the official BN VGG16 checkpoints.
"""
from torch import nn


class VGG16Cifar(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.init_channels = 3
        self.T = 0
        self.layer1 = self._make_layers((64, 64))
        self.layer2 = self._make_layers((128, 128))
        self.layer3 = self._make_layers((256, 256, 256))
        self.layer4 = self._make_layers((512, 512, 512))
        self.layer5 = self._make_layers((512, 512, 512))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(512, 4096), nn.ReLU(inplace=False), nn.Dropout(0.),
            nn.Linear(4096, 4096), nn.ReLU(inplace=False), nn.Dropout(0.),
            nn.Linear(4096, num_classes),
        )
        # Keep the author's construction and initialization order. In particular,
        # Conv biases retain their default initialization; Linear biases are zero.
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)

    def _make_layers(self, channels):
        layers = []
        for width in channels:
            layers.extend((nn.Conv2d(self.init_channels, width, 3, padding=1),
                           nn.BatchNorm2d(width), nn.ReLU(inplace=False), nn.Dropout(0.)))
            self.init_channels = width
        layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, images):
        x = images
        for stage in range(1, 6):
            x = getattr(self, f'layer{stage}')(x)
        return self.classifier(x)


def vgg16_cifar_ann(num_classes=100, base_width=64):
    """The author architecture has fixed widths; no configurable narrow head."""
    if base_width != 64:
        raise ValueError('Author CIFAR VGG16 uses fixed base width 64')
    return VGG16Cifar(num_classes=num_classes)
