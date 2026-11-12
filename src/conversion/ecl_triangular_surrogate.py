"""Ordinary normalized triangular surrogate for unchanged signed ECL T2.

Both threshold comparisons retain the deployment rule. Only their backward
uses max(1 - abs(z), 0), where z is distance in threshold units. Retraction
eligibility is a hard count gate; the physical reset remains differentiable.
This is a conventional surrogate, not an exact hard-network derivative.
"""
from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

from src.conversion.qcfs_ecl_resnet18 import DualThresholdSequence, ECLResNet18


class _TriangularEvent(torch.autograd.Function):
    @staticmethod
    def forward(ctx, membrane, threshold, eligible):
        # Compare the original membrane, not a rounded normalized margin.
        event = (membrane > threshold).to(membrane.dtype)
        event = event - ((membrane <= -1e-3) & eligible).to(membrane.dtype)
        if ctx.needs_input_grad[0]:
            positive_margin = (membrane - threshold) / threshold
            negative_margin = (-1e-3 - membrane) / threshold
            positive_slope = (1 - positive_margin.abs()).clamp_min(0)
            negative_slope = (1 - negative_margin.abs()).clamp_min(0)
            # e = H(m-theta) - gate*H(-epsilon-m): the outer subtraction
            # and the negative margin's derivative cancel for retraction.
            slope = (positive_slope + eligible * negative_slope) / threshold
            ctx.save_for_backward(slope)
        return event

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        if not ctx.needs_input_grad[0]:
            return None, None, None
        (slope,) = ctx.saved_tensors
        return gradient * slope, None, None


class TriangularSequence(DualThresholdSequence):
    """Ordinary ECL events with a fixed unit-halfwidth triangular VJP."""

    def __init__(self, threshold):
        super().__init__(threshold, timesteps=2, signed=True)

    def forward(self, x):
        if not torch.is_grad_enabled():
            return super().forward(x)
        values = x.unflatten(0, (self.timesteps, -1))
        membrane = .5 * self.threshold
        count = torch.zeros_like(values[0])
        result = []
        for current in values:
            membrane = membrane + current
            # Boolean comparison has no VJP; no derivative through eligibility.
            event = _TriangularEvent.apply(membrane, self.threshold, count >= 1)
            output = event * self.threshold
            membrane = membrane - output
            count = count + event
            result.append(output)
        return torch.stack(result).flatten(0, 1)


class TriangularResNet18(ECLResNet18):
    """Same deployment graph and state_dict keys as ECLResNet18."""

    def __init__(self, source):
        super().__init__(source, timesteps=2, signed=True)
        for name, node in list(self.source.named_modules()):
            if isinstance(node, DualThresholdSequence):
                parent, _, child = name.rpartition('.')
                setattr(self.source.get_submodule(parent), child,
                        TriangularSequence(node.threshold))
        # Thresholds remain buffers; original Conv/BN affine/FC are trainable.
        self.requires_grad_(True)
