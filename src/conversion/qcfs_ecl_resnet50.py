"""Channelwise QCFS-L2 and retained signed-ECL/phase conversion for CIFAR R50.

Only the architecture traversal differs from the existing ResNet18 path:
stem plus all three activations of each of the sixteen Bottleneck blocks.
The author QCFS operator, physical ECL events and retained T2 phase surrogate
are reused unchanged. Converted thresholds are fixed buffers.
"""
from __future__ import annotations

import copy

import torch
from torch import nn

from src.conversion.channel_qcfs_source import (
    MIN_THRESHOLD, project_thresholds, threshold_parameters,
)
from src.conversion.joint_phase_adjoint import JointPhaseSequence
from src.conversion.qcfs_ecl_resnet18 import AuthorQCFS, DualThresholdSequence, ECLResNet18


def activation_sites(source):
    """Yield all49 (module path, channel count) pairs, including every relu3."""
    if tuple(len(getattr(source, f'layer{stage}')) for stage in range(1, 5)) != (3, 4, 6, 3):
        raise ValueError('Expected the CIFAR ResNet50 Bottleneck layout (3,4,6,3)')
    yield 'relu', source.bn1.num_features
    for stage in range(1, 5):
        for index, block in enumerate(getattr(source, f'layer{stage}')):
            for k in (1, 2, 3):
                name = f'layer{stage}.{index}.relu{k}'
                source.get_submodule(name)
                yield name, getattr(block, f'bn{k}').num_features


def install_channel_qcfs(source, units):
    """Install original QCFS-L2 with direct positive per-channel parameters."""
    sites = list(activation_sites(source))
    if set(units) != {name for name, _ in sites}:
        raise ValueError('Calibration must supply all49 activations exactly once')
    thresholds = {}
    for name, channels in sites:
        threshold = units[name].detach().clone().reshape(1, -1, 1, 1)
        if threshold.numel() != channels:
            raise ValueError(f'{name} requires {channels} channel thresholds')
        if not bool(torch.isfinite(threshold).all() and (threshold > 0).all()):
            raise ValueError('QCFS thresholds must be finite and positive')
        thresholds[name] = threshold.to(source.conv1.weight).clamp_min(MIN_THRESHOLD)
    for name, _ in sites:
        node = AuthorQCFS(T=0, L=2, thresh=1.).to(source.conv1.weight)
        node.thresh = nn.Parameter(thresholds[name])
        parent, _, child = name.rpartition('.')
        setattr(source.get_submodule(parent), child, node)
    return source


class ECLResNet50(ECLResNet18):
    """Ordinary physical ECL with R50 traversal and the existing packed forward."""
    def __init__(self, source, timesteps=2, signed=True):
        nn.Module.__init__(self)
        self.source = copy.deepcopy(source).eval().requires_grad_(False)
        self.timesteps = timesteps
        for name, _ in activation_sites(source):
            previous = self.source.get_submodule(name)
            if not isinstance(previous, AuthorQCFS) or previous.T != 0 or previous.L != 2:
                raise ValueError('Conversion requires a QCFS-L2 ANN source')
            parent, _, child = name.rpartition('.')
            setattr(self.source.get_submodule(parent), child,
                    DualThresholdSequence(previous.thresh, timesteps, signed))
        self.eval().requires_grad_(False)


class JointPhaseResNet50(ECLResNet50):
    """Unchanged T2 ECL forward, retained local phase VJP at all49 sites."""
    def __init__(self, source):
        super().__init__(source, timesteps=2, signed=True)
        for name, _ in activation_sites(self.source):
            previous = self.source.get_submodule(name)
            parent, _, child = name.rpartition('.')
            setattr(self.source.get_submodule(parent), child,
                    JointPhaseSequence(previous.threshold))
        self.requires_grad_(True)
