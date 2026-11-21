"""DiMP18 feature-backbone preparation; native tracking heads stay unchanged.

Official weights must be loaded before splitting repeated ReLU calls. Each
image owns its own T-step trajectory; template and search never share state.
Only source quantization L and inference horizon T define the converted model.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import sys

import torch
from torch import nn

from src.conversion.activation_sites import activation_sites
from src.conversion.ecl_triangular_surrogate import TriangularSequence
from src.conversion.qcfs_ecl_resnet18 import AuthorQCFS, DualThresholdSequence
from src.conversion.triangular_density_horizon import TriangularDensityHorizonSequence


def load_official_dimp18(upstream, checkpoint):
    """Strict-load the published full model, without downloading other weights."""
    upstream, checkpoint = Path(upstream).resolve(), Path(checkpoint).resolve()
    if not (upstream / 'ltr/models/tracking/dimpnet.py').is_file():
        raise ValueError('Expected the prepared official PyTracking checkout')
    if 'ltr' in sys.modules:
        loaded = Path(sys.modules['ltr'].__file__).resolve()
        if upstream not in loaded.parents:
            raise RuntimeError('Another PyTracking checkout is already imported')
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    import ltr.admin.loading as loading
    loading._setup_legacy_env()
    try:
        raw = torch.load(checkpoint, map_location='cpu', weights_only=False)
    finally:
        loading._cleanup_legacy_env()
    constructor = copy.deepcopy(raw['constructor'])
    if constructor.fun_name != 'dimpnet18':
        raise ValueError('Expected the official DiMP18 constructor')
    constructor.kwds['backbone_pretrained'] = False
    model = constructor.get()
    model.load_state_dict(raw['net'], strict=True)
    if any(not bool(torch.isfinite(t).all()) for t in model.state_dict().values()
           if t.is_floating_point()):
        raise ValueError('Non-finite official checkpoint')
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return model.eval(), dict(checkpoint=str(checkpoint), sha256=digest,
                              constructor='dimpnet18', strict_load=True)


class _IndependentBasicBlock(nn.Module):
    """ANN-equivalent block with separate activation sites for both calls."""
    def __init__(self, original):
        super().__init__()
        if not isinstance(original.relu, nn.ReLU) or not original.use_bn:
            raise ValueError('Expected the author BN/BasicBlock ResNet18')
        for name in ('conv1', 'bn1', 'conv2', 'bn2', 'downsample'):
            setattr(self, name, getattr(original, name))
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)
        self.stride = original.stride

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu1(self.bn1(self.conv1(x)))
        return self.relu2(self.bn2(self.conv2(out)) + residual)


def split_dimp18_activations(backbone):
    """Copy an already loaded backbone, preserving all learned tensor keys."""
    result = copy.deepcopy(backbone)
    for stage in range(1, 4):
        layer = getattr(result, f'layer{stage}')
        for index, block in enumerate(layer):
            layer[index] = _IndependentBasicBlock(block)
    # DiMP18 consumes only layer2 and layer3; layer4/fc are never executed.
    result.num_stages = 3
    for name in ('layer4', 'fc'):
        if hasattr(result, name):
            getattr(result, name).requires_grad_(False)
    return result.eval()


@torch.no_grad()
def calibrate_dimp_thresholds(backbone, image_batches):
    """Observe activation maxima; this is initialization, not a trained source."""
    maxima = {name: 0.0 for name, _ in activation_sites(backbone)}
    handles = []
    for name in maxima:
        node = backbone.get_submodule(name)
        if not isinstance(node, nn.ReLU):
            raise ValueError('Calibration expects the split ReLU backbone')
        def observe(_module, _inputs, output, key=name):
            value = float(output.detach().amax())
            if not bool(torch.isfinite(output).all()):
                raise ValueError('Non-finite calibration activation')
            maxima[key] = max(maxima[key], value)
        handles.append(node.register_forward_hook(observe))
    count = 0
    try:
        backbone.eval()
        for images in image_batches:
            backbone(images, ['layer2', 'layer3'])
            count += len(images)
    finally:
        for handle in handles:
            handle.remove()
    if count == 0:
        raise ValueError('At least one real calibration image is required')
    dead = [name for name, value in maxima.items() if value <= 0]
    return dict(thresholds={name: value if value > 0 else 1.0
                            for name, value in maxima.items()},
                images=count, statistic='observed_activation_max',
                inactive_sites_with_unit_initialization=dead,
                status='initialization_only_requires_task_source_training')


def install_dimp_qcfs(backbone, L, thresholds):
    """Copy split backbone and install the pinned author's QCFS ANN nodes."""
    if isinstance(L, bool) or not isinstance(L, int) or L < 1:
        raise ValueError('L must be a positive integer')
    result = copy.deepcopy(backbone)
    sites = dict(activation_sites(result))
    if set(thresholds) != set(sites):
        raise ValueError('Thresholds must match every active activation site')
    for name in sites:
        value = float(thresholds[name])
        if not value > 0 or not bool(torch.isfinite(torch.tensor(value))):
            raise ValueError('Expected finite positive thresholds')
        parent, _, child = name.rpartition('.')
        setattr(result.get_submodule(parent), child,
                AuthorQCFS(T=0, L=L, thresh=value).to(result.conv1.weight))
    return result.eval()


class _CumulativeMaxPool(nn.Module):
    """Pool cumulative charge and emit differences, with fresh image state."""
    def __init__(self, pool, T):
        super().__init__()
        self.pool, self.T = pool, T

    def forward(self, x):
        ticks = x.unflatten(0, (self.T, -1))
        cumulative = ticks.cumsum(0)
        pooled = self.pool(cumulative.flatten(0, 1)).unflatten(0, (self.T, -1))
        return torch.cat((pooled[:1], pooled[1:] - pooled[:-1]), 0).flatten(0, 1)


class DimpFeatureBackbone(nn.Module):
    """Drop-in dictionary feature interface for source or converted DiMP18.

    The analog DiMP classifier/IoU heads receive decoded layer2/layer3 maps.
    No second full prediction or template/search averaging is performed.
    """
    def __init__(self, qcfs_source, T=4, method='density'):
        super().__init__()
        if isinstance(T, bool) or not isinstance(T, int) or T < 1:
            raise ValueError('T must be a positive integer')
        if method not in ('source', 'hard', 'triangular', 'density'):
            raise ValueError('Unknown DiMP conversion method')
        self.source = copy.deepcopy(qcfs_source)
        self.T, self.method = T, method
        levels = set()
        for name, _ in list(activation_sites(self.source)):
            node = self.source.get_submodule(name)
            if not isinstance(node, AuthorQCFS) or node.T != 0:
                raise ValueError('Expected a split QCFS ANN source')
            levels.add(node.L)
            if method == 'source':
                continue
            if method == 'density':
                replacement = TriangularDensityHorizonSequence(node.thresh, T)
            elif method == 'triangular':
                replacement = TriangularSequence(node.thresh)
                replacement.timesteps = T
            else:
                replacement = DualThresholdSequence(node.thresh, T, signed=True)
            parent, _, child = name.rpartition('.')
            setattr(self.source.get_submodule(parent), child, replacement)
        if len(levels) != 1:
            raise ValueError('All quantized sites must share one source L')
        self.L = levels.pop()
        if method != 'source':
            self.source.maxpool = _CumulativeMaxPool(self.source.maxpool, T)
        self.eval()

    def train(self, mode=True):
        super().train(mode)
        # Running statistics stay at the common source values. Affine weights
        # may still receive gradients during later matched task fine-tuning.
        for module in self.source.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def forward(self, images, output_layers=None):
        layers = ['layer2', 'layer3'] if output_layers is None else list(output_layers)
        if not layers or not set(layers) <= {'conv1', 'layer1', 'layer2', 'layer3'}:
            raise ValueError('DiMP18 conversion supports features through layer3')
        if self.method == 'source':
            return self.source(images, layers)
        # Temporal-major layout keeps every image's sequence independent.
        expanded = images.unsqueeze(0).expand(self.T, *images.shape).flatten(0, 1)
        features = self.source(expanded, layers)
        return type(features)((name, feature.unflatten(0, (self.T, len(images))).mean(0))
                              for name, feature in features.items())

    def out_feature_channels(self, layer=None):
        return self.source.out_feature_channels(layer)

    def out_feature_strides(self, layer=None):
        return self.source.out_feature_strides(layer)
