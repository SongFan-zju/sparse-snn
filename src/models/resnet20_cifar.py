"""Author-compatible CIFAR ResNet20: 3x3 BasicBlocks, widths 16/32/64.

Projection shortcuts match QCFS's ResNet4Cifar (not option-A zero padding).
An identity fourth stage preserves the existing temporal forward interface.
PyTorch module default initialization matches the author implementation.
"""
from torch import nn
from src.models.resnet18_cifar import ResNet18Cifar, BasicBlock, _conv3x3

class ResNet20Cifar(ResNet18Cifar):
    def __init__(self,num_classes=100,base_width=16):
        nn.Module.__init__(self)
        self.num_stages=3
        self.in_channels=base_width
        self.conv1=_conv3x3(3,base_width)
        self.bn1=nn.BatchNorm2d(base_width)
        self.relu=nn.ReLU(inplace=False)
        self.layer1=self._make_layer(base_width,1)
        self.layer2=self._make_layer(base_width*2,2)
        self.layer3=self._make_layer(base_width*4,2)
        self.layer4=nn.Identity()
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(base_width*4,num_classes)

    def _make_layer(self,channels,stride):
        first=BasicBlock(self.in_channels,channels,stride)
        self.in_channels=channels
        return nn.Sequential(first,BasicBlock(channels,channels),BasicBlock(channels,channels))

def resnet20_cifar_ann(num_classes=100,base_width=16):
    return ResNet20Cifar(num_classes,base_width)
