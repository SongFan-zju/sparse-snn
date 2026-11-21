"""Hard signed ECL T2 with a triangular-density shared-phase mean VJP.

Only backward changes: integrate the complete two-event trajectory under
rho(phi) = max(1 - abs(phi - 1/2), 0), supported on [-1/2, 3/2]. This is a
hypothetical smoothing distribution, including initial states outside [0, 1],
not a model of physical initial membrane uncertainty. Deployment still starts
at theta/2 and executes one unchanged hard trajectory.

The local Jacobian is exact for this conditional charge-mean map. Combining
it with hard-network output adjoints remains a biased surrogate; it is not
an exact deployment-loss or unbiased whole-network stochastic-loss gradient.
"""
from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

from src.conversion.joint_phase_adjoint import hard_pair
from src.conversion.qcfs_ecl_resnet18 import DualThresholdSequence, ECLResNet18


def triangular_density_cdf(value):
    """CDF of the fixed normalized initial phase, with mode 1/2 and radius 1."""
    # Clamp before squaring: even unused branches stay finite for large inputs.
    bounded = value.clamp(-.5, 1.5)
    return torch.where(bounded <= .5,
                       .5 * (bounded + .5).square(),
                       1 - .5 * (1.5 - bounded).square())


def triangular_density_charge_means(current0, current1, threshold):
    """Return E[theta*s0], E[theta*s1] under one shared triangular phase.

    Incoming currents and the positive threshold are fixed conditioning
    variables; s1 includes reset and eligible retraction with epsilon=.001.
    These are charge means, so callers must not multiply by threshold again.
    """
    first_boundary = 1 - current0 / threshold
    total = (current0 + current1) / threshold
    first_cdf = triangular_density_cdf(first_boundary)
    first = 1 - first_cdf
    positive_after_zero = (
        first_cdf - triangular_density_cdf(1 - total)).clamp_min(0)
    positive_after_one = 1 - triangular_density_cdf(
        torch.maximum(first_boundary, 2 - total))
    negative_after_one = (
        triangular_density_cdf(1 - total - 1e-3 / threshold)
        - first_cdf).clamp_min(0)
    return threshold * first, threshold * (
        positive_after_zero + positive_after_one - negative_after_one)


class TriangularDensityPhaseEvent(torch.autograd.Function):
    @staticmethod
    def forward(ctx, current0, current1, threshold):
        ctx.save_for_backward(current0, current1, threshold)
        return hard_pair(current0, current1, threshold)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad0, grad1):
        current0, current1, threshold = ctx.saved_tensors
        with torch.enable_grad():
            first = current0.detach().requires_grad_(True)
            second = current1.detach().requires_grad_(True)
            means = triangular_density_charge_means(first, second, threshold)
            gradients = torch.autograd.grad(
                means, (first, second), grad_outputs=(grad0, grad1),
                create_graph=False)
        return gradients[0], gradients[1], None


class TriangularDensityPhaseSequence(DualThresholdSequence):
    def __init__(self, threshold):
        super().__init__(threshold, timesteps=2, signed=True)

    def forward(self, x):
        if not torch.is_grad_enabled():
            return super().forward(x)
        first, second = x.unflatten(0, (2, -1)).unbind(0)
        return torch.stack(TriangularDensityPhaseEvent.apply(
            first, second, self.threshold)).flatten(0, 1)


class TriangularDensityPhaseResNet18(ECLResNet18):
    """ECL-compatible state_dict and forward; also accepts standard CIFAR R20."""
    def __init__(self, source):
        super().__init__(source, timesteps=2, signed=True)
        for name, node in list(self.source.named_modules()):
            if isinstance(node, DualThresholdSequence):
                parent, _, child = name.rpartition('.')
                setattr(self.source.get_submodule(parent), child,
                        TriangularDensityPhaseSequence(node.threshold))
        self.requires_grad_(True)
