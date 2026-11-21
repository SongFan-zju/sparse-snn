"""QCFS author-compatible VGG16 ANN and standard binary signed-ECL conversion.

All 15 activation sites, including both hidden FC activations, run temporally.
The final affine classifier runs once on time-mean hidden features. No model
fusion, repeat inference, or event-capacity variant is introduced.
"""
from __future__ import annotations

import copy

import torch
from torch import nn

from src.conversion.general_phase_adjoint import _positive_integer, TriangularHorizonSequence
from src.conversion.joint_phase_adjoint import JointPhaseSequence
from src.conversion.qcfs_ecl_resnet18 import AuthorQCFS, DualThresholdSequence
from src.conversion.recursive_phase_adjoint import RecursivePhaseSequence
from src.conversion.triangular_density_horizon import TriangularDensityHorizonSequence
from src.conversion.triangular_density_phase import TriangularDensityPhaseSequence


def activation_sites(source):
    """Yield author activation names and widths, without ResNet enumeration."""
    for stage in range(1, 6):
        name = f'layer{stage}'
        for index, module in enumerate(getattr(source, name)):
            if isinstance(module, nn.BatchNorm2d):
                yield f'{name}.{index + 1}', module.num_features
    yield 'classifier.2', source.classifier[1].out_features
    yield 'classifier.5', source.classifier[4].out_features


def _activation_threshold(value, width, fully_connected):
    threshold = value.detach().clone()
    if threshold.numel() == 1:
        # Preserve the official scalar parameter shape (normally [1]). A FC
        # activation receives a 2D tensor and cannot broadcast a 4D threshold.
        if threshold.ndim > (2 if fully_connected else 4):
            raise ValueError('Scalar threshold has too many dimensions for this activation')
    elif threshold.numel() == width:
        threshold = threshold.reshape(1, width) if fully_connected else threshold.reshape(1, width, 1, 1)
    else:
        raise ValueError('Threshold must be scalar or supply one value per activation channel')
    if not bool(torch.isfinite(threshold).all() and (threshold > 0).all()):
        raise ValueError('QCFS thresholds must be finite and positive')
    return threshold


def install_qcfs(source, levels=2, thresholds=None):
    """Install T=0 QCFS with scalar or correctly broadcast channel thresholds.

    Official scalar shapes remain unchanged for strict checkpoint loading.
    Convolutional channel thresholds use [1,C,1,1]; FC thresholds use [1,C].
    """
    _positive_integer(levels, 'levels')
    sites = list(activation_sites(source))
    if thresholds is not None and set(thresholds) != {name for name, _ in sites}:
        raise ValueError('Thresholds must supply each VGG activation exactly once')
    weight = source.layer1[0].weight
    for name, width in sites:
        old = source.get_submodule(name)
        if thresholds is None:
            value = old.thresh if isinstance(old, AuthorQCFS) else weight.new_tensor([8.])
        else:
            value = thresholds[name]
        threshold = _activation_threshold(value, width, name.startswith('classifier.')).to(weight)
        activation = AuthorQCFS(T=0, L=levels, thresh=8.).to(weight)
        activation.thresh = nn.Parameter(threshold)
        parent, _, child = name.rpartition('.')
        setattr(source.get_submodule(parent), child, activation)
    return source


def install_channel_qcfs(source, thresholds, levels=2):
    """Factory-compatible calibration entry point; also preserves scalar shapes."""
    return install_qcfs(source, levels=levels, thresholds=thresholds)


def _sequence(threshold, timesteps, method):
    if method == 'hard':
        return DualThresholdSequence(threshold, timesteps=timesteps, signed=True)
    if method == 'phase':
        return JointPhaseSequence(threshold) if timesteps == 2 else RecursivePhaseSequence(threshold, timesteps)
    if method == 'triangular':
        return TriangularHorizonSequence(threshold, timesteps)
    if method == 'phase_triangular_density':
        return (TriangularDensityPhaseSequence(threshold) if timesteps == 2
                else TriangularDensityHorizonSequence(threshold, timesteps))
    raise ValueError(f'Unknown VGG ECL gradient method: {method}')


class ECLVGG16(nn.Module):
    """Unchanged signed-ECL forward with a selectable training-only local VJP.

    Source ANN activations must be QCFS at T=0. Thresholds become fixed buffers;
    method='hard' freezes parameters, while the gradient variants train Conv,
    BN affine, and all three FC layers. All variants have identical state keys.
    """
    def __init__(self, source, timesteps=2, method='hard'):
        super().__init__()
        _positive_integer(timesteps, 'timesteps')
        if method not in ('hard', 'phase', 'triangular', 'phase_triangular_density'):
            raise ValueError(f'Unknown VGG ECL gradient method: {method}')
        if getattr(source, 'T', 0) != 0:
            raise ValueError('Conversion requires a T=0 QCFS ANN source')
        self.source = copy.deepcopy(source).eval().requires_grad_(False)
        self.timesteps = timesteps
        for name, _ in list(activation_sites(self.source)):
            previous = self.source.get_submodule(name)
            if not isinstance(previous, AuthorQCFS) or previous.T != 0:
                raise ValueError('Conversion requires all 15 VGG activations to be T=0 QCFS')
            parent, _, child = name.rpartition('.')
            setattr(self.source.get_submodule(parent), child,
                    _sequence(previous.thresh, timesteps, method))
        self.eval().requires_grad_(method != 'hard')

    def forward(self, images):
        x = images.unsqueeze(0).expand(self.timesteps, *images.shape).flatten(0, 1)
        for stage in range(1, 6):
            x = getattr(self.source, f'layer{stage}')(x)
        for index in range(len(self.source.classifier) - 1):
            x = self.source.classifier[index](x)
        features = x.unflatten(0, (self.timesteps, len(images))).mean(0)
        return self.source.classifier[-1](features)
