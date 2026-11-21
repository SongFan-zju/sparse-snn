"""Author Fast-SNN ResNet18 classification graph with interchangeable VJPs.

The upstream ResNet18 uses stages [3,3,2], widths128/256/512 and a64-channel
stem. It differs from the four-stage ResNet18 used elsewhere in this project.
Classification retains the author's >= firing comparison and summed logits.
"""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import torch
from torch import nn
from torch.autograd.function import once_differentiable

from src.conversion.triangular_density_horizon import triangle_density_partition_vjp

_RUNTIME = None


def author_runtime(path):
    global _RUNTIME
    path = Path(path).resolve()
    if _RUNTIME is not None:
        if _RUNTIME.path != path:
            raise ValueError('Only one author classification checkout per process')
        return _RUNTIME
    prefix = '_sparse_fast_classification'
    for suffix in ['', '.models']:
        package = ModuleType(prefix + suffix)
        package.__path__ = [str(path / suffix.lstrip('.'))]
        sys.modules[package.__name__] = package
    name = prefix + '.models.quant_layer'
    spec = importlib.util.spec_from_file_location(name, path / 'models/quant_layer.py')
    quant = importlib.util.module_from_spec(spec)
    sys.modules[name] = quant
    spec.loader.exec_module(quant)
    name = prefix + '.models.resnet'
    model = ModuleType(name)
    model.__file__ = str(path / 'models/resnet.py')
    sys.modules[name] = model
    source = Path(model.__file__).read_text().replace(
        'from models.quant_layer import QuantReLU',
        f'from {prefix}.models.quant_layer import QuantReLU')
    exec(compile(source, model.__file__, 'exec'), model.__dict__)
    _RUNTIME = SimpleNamespace(path=path, quant=quant, model=model)
    return _RUNTIME


def make_ann(runtime, bit=32):
    return runtime.model.resnet18(bit=bit, num_classes=100)


@torch.no_grad()
def author_events(currents, threshold):
    """The original ordered positive reset, count update, then withdrawal."""
    membrane = .5 * threshold
    count = torch.zeros_like(currents[0])
    outputs = []
    for current in currents:
        membrane = membrane + current
        positive = (membrane >= threshold).to(current.dtype)
        membrane = membrane - positive * threshold
        count = count + positive
        negative = ((membrane <= -1e-3) & (count > 0)).to(current.dtype)
        membrane = membrane + negative * threshold
        count = count - negative
        outputs.append((positive - negative) * threshold)
    return torch.stack(outputs)


class _FastTriangle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, membrane, threshold, eligible):
        positive = (membrane >= threshold).to(membrane.dtype)
        negative = ((membrane <= -1e-3) & eligible).to(membrane.dtype)
        slope = ((1 - ((membrane-threshold)/threshold).abs()).clamp_min(0)
                 + eligible * (1 - ((-1e-3-membrane)/threshold).abs()).clamp_min(0))/threshold
        ctx.save_for_backward(slope)
        return positive - negative

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        return gradient * ctx.saved_tensors[0], None, None


class _FastDensity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, currents, threshold):
        ctx.save_for_backward(currents, threshold)
        return author_events(currents, threshold)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        currents, threshold = ctx.saved_tensors
        return triangle_density_partition_vjp(currents, threshold, gradient), None


class FastSignedSequence(nn.Module):
    def __init__(self, threshold, T, method):
        super().__init__()
        if T < 1 or method not in ('hard', 'triangular', 'density'):
            raise ValueError('Invalid Fast-SNN T or VJP')
        if not bool(torch.isfinite(threshold).all() and (threshold > 0).all()):
            raise ValueError('Positive finite source thresholds required')
        self.register_buffer('threshold', threshold.detach().clone())
        self.T, self.method = T, method

    def forward(self, x):
        currents = x.unflatten(0, (-1, self.T)).transpose(0, 1)
        if not torch.is_grad_enabled() or self.method == 'hard':
            output = author_events(currents, self.threshold)
        elif self.method == 'density':
            output = _FastDensity.apply(currents, self.threshold)
        else:
            membrane = .5*self.threshold
            count = torch.zeros_like(currents[0])
            outputs = []
            for current in currents:
                membrane = membrane + current
                event = _FastTriangle.apply(membrane, self.threshold, count >= 1)
                charge = event * self.threshold
                membrane = membrane - charge
                count = count + event
                outputs.append(charge)
            output = torch.stack(outputs)
        return output.transpose(0, 1).flatten(0, 1)


class FastClassifier(nn.Module):
    def __init__(self, source, runtime, T, method):
        super().__init__()
        self.source = copy.deepcopy(source)
        self.T = T
        self.method = method
        sites = [(name, module) for name, module in self.source.named_modules()
                 if isinstance(module, runtime.quant.QuantReLU)]
        if len(sites) != 17:
            raise ValueError('Author ResNet18 has17 quantized activation sites')
        for name, module in sites:
            parent, _, child = name.rpartition('.')
            setattr(self.source.get_submodule(parent), child,
                    FastSignedSequence(module.act_alpha, T, method))
        self.requires_grad_(method != 'hard')

    def train(self, mode=True):
        super().train(mode)
        # Native converted inference uses source running statistics. Both SG
        # arms keep these fixed; BN affine, convolutions and classifier train.
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def forward(self, x):
        batch = len(x)
        packed = x.unsqueeze(1).expand(batch, self.T, *x.shape[1:]).flatten(0, 1)
        return self.source(packed).unflatten(0, (batch, self.T)).sum(1)
