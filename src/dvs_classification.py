"""Shared event preprocessing and ResNet conversion for the DVS comparison.

This is a local adaptation of the static-image QCFS/ECL conversion, not an
author-published DVS recipe. One event bin is consumed per SNN timestep.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from src.models.resnet18_cifar import ResNet18Cifar
from src.models.resnet20_cifar import ResNet20Cifar
from src.conversion.activation_sites import activation_sites
from src.conversion.qcfs_ecl_resnet18 import AuthorQCFS, DualThresholdSequence
from src.conversion.ecl_triangular_surrogate import TriangularSequence
from src.conversion.triangular_density_horizon import TriangularDensityHorizonSequence


class DVSFrames(Dataset):
    """A frozen manifest and shared spatial augmentation of all event bins.

    Normalize by each sample's pooled histogram mean. The ANN observes the
    integrated histogram; the SNN observes chronological currents whose mean
    is exactly that histogram. No repeated static-image inference or fusion.
    """
    def __init__(self, manifest, split, training=False):
        self.manifest = Path(manifest)
        data = json.loads(self.manifest.read_text())
        self.dataset_name = data.get('dataset','CIFAR10-DVS')
        self.official_split = bool(data['official_split'])
        expected = {'CIFAR10-DVS':False,'DVS128 Gesture':True}
        if data['status'] != 'complete' or expected.get(self.dataset_name) != self.official_split:
            raise ValueError('Unsupported event dataset or incorrect split policy')
        self.num_classes = len(data['classes'])
        if self.num_classes != (11 if self.dataset_name=='DVS128 Gesture' else 10):
            raise ValueError('Event class count differs from the dataset')
        self.horizontal_flip = self.dataset_name=='CIFAR10-DVS'
        if split not in ('train', 'test'):
            raise ValueError('Expected train or test split')
        self.samples = [row for row in data['samples'] if row['split'] == split]
        self.T = data['T']
        self.training = training
        self.frames_cache = None

    def preload(self):
        """Keep pooled counts in CPU RAM, avoiding repeated archive decoding."""
        self.frames_cache = torch.empty(len(self), self.T, 2, 48, 48)
        for index in range(len(self)):
            self.frames_cache[index].copy_(self._read_frames(index))
        return self

    def __len__(self):
        return len(self.samples)

    def _read_frames(self, index):
        row = self.samples[index]
        with np.load(self.manifest.parent / row['path']) as item:
            frames = torch.from_numpy(item['frames'].astype(np.float32))
        # Resolution is a fixed preprocessing choice, shared by every method.
        return F.interpolate(frames, size=(48, 48), mode='area')

    def __getitem__(self, index):
        row = self.samples[index]
        frames = (self._read_frames(index) if self.frames_cache is None
                  else self.frames_cache[index])
        if self.training:
            frames = F.pad(frames, (4, 4, 4, 4))
            top, left = torch.randint(9, (2,)).tolist()
            frames = frames[:, :, top:top+48, left:left+48]
            if self.horizontal_flip and torch.rand(()) < .5:
                frames = frames.flip(-1)
        histogram = frames.sum(0)
        scale = histogram.mean().clamp_min(1.e-6)
        currents = frames * (self.T / scale)
        return currents, row['target']


def make_ann(architecture, num_classes=10):
    if architecture == 'resnet18':
        model = ResNet18Cifar(num_classes=num_classes)
    elif architecture == 'resnet20':
        model = ResNet20Cifar(num_classes=num_classes)
    else:
        raise ValueError(f'Unsupported DVS backbone: {architecture}')
    model.conv1 = nn.Conv2d(2, model.conv1.out_channels, 3, padding=1, bias=False)
    nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
    return model


def install_qcfs(model, L, thresholds=None):
    for name, _ in list(activation_sites(model)):
        node = AuthorQCFS(T=0, L=L, thresh=8.).to(model.conv1.weight)
        if thresholds is not None:
            node.thresh.data.copy_(thresholds[name].to(node.thresh))
        parent, _, child = name.rpartition('.')
        setattr(model.get_submodule(parent), child, node)
    return model


class EventResNet(nn.Module):
    def __init__(self, source, T, method):
        super().__init__()
        if method not in ('qcfs', 'ecl', 'triangular', 'density'):
            raise ValueError(f'Unsupported conversion: {method}')
        self.source = copy.deepcopy(source)
        self.T = T
        for name, _ in list(activation_sites(source)):
            previous = source.get_submodule(name)
            if not isinstance(previous, AuthorQCFS) or previous.T:
                raise ValueError('Conversion requires a QCFS source ANN')
            if method == 'density':
                node = TriangularDensityHorizonSequence(previous.thresh, T)
            elif method == 'triangular':
                node = TriangularSequence(previous.thresh)
                node.timesteps = T
            else:
                node = DualThresholdSequence(previous.thresh, T, signed=method == 'ecl')
            parent, _, child = name.rpartition('.')
            setattr(self.source.get_submodule(parent), child, node)
        self.requires_grad_(method in ('triangular', 'density'))
        self.eval()

    def train(self, mode=True):
        super().train(mode)
        # Keep the common source BN buffers fixed for the SNN comparison.
        # Affine parameters remain trainable, identically in both gradient arms.
        for module in self.source.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def forward(self, currents):
        if currents.ndim != 5 or currents.shape[1] != self.T:
            raise ValueError('Expected [batch, T, polarity, height, width] event currents')
        batch = currents.shape[0]
        x = currents.transpose(0, 1).flatten(0, 1)
        net = self.source
        x = net.relu(net.bn1(net.conv1(x)))
        x = net.layer4(net.layer3(net.layer2(net.layer1(x))))
        features = net.avgpool(x).flatten(1).unflatten(0, (self.T, batch))
        return net.fc(features.mean(0))
