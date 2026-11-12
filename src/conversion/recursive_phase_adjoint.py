"""O(T) conditional shared-phase mean Jacobian, with unchanged hard ECL.

For fixed incoming currents, fixed positive threshold theta, no leakage, and
one signed event per step, the cumulative count is floor(phi + R_t/theta)
almost everywhere in a uniform normalized initial phase phi in [0, 1).
The expected emitted charge is therefore R_t - R_(t-1). This recurrence
computes that mean and its local input VJP without enumerating phase cells.

This is a local surrogate for the deterministic network, not its exact gradient
or an unbiased gradient of a whole-network stochastic loss. At branch ties we
choose hold, positive saturation, or negative maximum priority 0 > R-theta >
S+epsilon. This differs from the generic partition's zero-jump convention at
collisions/modulo endpoints. No phase-boundary sorting or cumsum is needed.
"""
from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

from src.conversion.general_phase_adjoint import hard_sequence
from src.conversion.qcfs_ecl_resnet18 import DualThresholdSequence, ECLResNet18


EPSILON = 1e-3


def _positive_timesteps(timesteps):
    if not isinstance(timesteps, int) or isinstance(timesteps, bool) or timesteps < 1:
        raise ValueError('timesteps must be a positive integer')


def _validate_inputs(currents, threshold):
    if currents.ndim < 2 or currents.shape[0] < 1 or not currents.is_floating_point():
        raise ValueError('Expected floating currents shaped [T, *neuron_shape]')
    shape = currents.shape[1:]
    if threshold.ndim > len(shape) or torch.broadcast_shapes(shape, threshold.shape) != shape:
        raise ValueError('Threshold must broadcast to the neuron shape')
    # The sequence constructor validates finite positive buffers once. Avoid
    # a device synchronization in every training forward.
    return threshold.to(device=currents.device, dtype=currents.dtype)


@torch.no_grad()
def _recurrence(currents, threshold, *, mean, coefficients):
    threshold = _validate_inputs(currents, threshold)
    total = torch.zeros_like(currents[0])
    charge = torch.zeros_like(total)
    means = torch.empty_like(currents) if mean else None
    a = torch.empty_like(currents, dtype=torch.bool) if coefficients else None
    b = torch.empty_like(currents, dtype=torch.bool) if coefficients else None
    for t, current in enumerate(currents):
        total = total + current
        positive = total > charge
        shifted = total + EPSILON
        negative = (~positive) & (shifted < charge)

        # Positive ties S == R+theta use the saturated branch R+theta.
        upper = charge + threshold
        positive_input = positive & (total < upper)
        # Negative maximum ties choose 0, then R-theta, then S+epsilon.
        lower = charge - threshold
        negative_zero = negative & (shifted <= 0) & (lower <= 0)
        negative_state = negative & (~negative_zero) & (lower >= shifted)
        negative_input = negative & (~negative_zero) & (~negative_state)
        next_charge = torch.where(
            positive, torch.where(positive_input, total, upper),
            torch.where(negative,
                        torch.where(negative_zero, 0.,
                                    torch.where(negative_state, lower, shifted)), charge))
        if coefficients:
            a[t] = ((~positive) & (~negative)) | (positive & (~positive_input)) | negative_state
            b[t] = positive_input | negative_input
        if mean:
            means[t] = next_charge - charge
        charge = next_charge
    return means, a, b


@torch.no_grad()
def recursive_phase_mean(currents, threshold):
    """Conditional emitted-charge means, with a fixed broadcast threshold."""
    return _recurrence(currents, threshold, mean=True, coefficients=False)[0]


@torch.no_grad()
def recursive_phase_coefficients(currents, threshold):
    """Boolean local derivatives a=dR_new/dR_old and b=dR_new/dS."""
    _, a, b = _recurrence(currents, threshold, mean=False, coefficients=True)
    return a, b


@torch.no_grad()
def _coefficient_vjp(a, b, gradient):
    result = torch.empty_like(gradient)
    adjoint = torch.zeros_like(gradient[0])
    suffix = torch.zeros_like(adjoint)
    for t in range(len(gradient) - 1, -1, -1):
        value = adjoint + gradient[t]
        total_gradient = torch.where(b[t], value, 0.)
        adjoint = torch.where(a[t], value, 0.) - gradient[t]
        suffix = suffix + total_gradient
        result[t] = suffix
    return result


@torch.no_grad()
def recursive_phase_vjp(currents, threshold, output_gradient):
    """VJP of conditional means w.r.t. currents; no threshold derivative."""
    if (output_gradient.shape != currents.shape or output_gradient.device != currents.device
            or output_gradient.dtype != currents.dtype):
        raise ValueError('Output gradient must match currents shape, device, and dtype')
    a, b = recursive_phase_coefficients(currents, threshold)
    return _coefficient_vjp(a, b, output_gradient)


class _RecursivePhaseEvent(torch.autograd.Function):
    @staticmethod
    def forward(ctx, currents, threshold):
        a, b = recursive_phase_coefficients(currents, threshold)
        # Only two boolean arrays survive forward, not currents or cell axes.
        ctx.save_for_backward(a, b)
        return hard_sequence(currents, threshold)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        a, b = ctx.saved_tensors
        return _coefficient_vjp(a, b, gradient), None


class RecursivePhaseSequence(DualThresholdSequence):
    def __init__(self, threshold, timesteps=4):
        _positive_timesteps(timesteps)
        super().__init__(threshold, timesteps=timesteps, signed=True)

    def forward(self, x):
        if not torch.is_grad_enabled():
            return super().forward(x)
        currents = x.unflatten(0, (self.timesteps, -1))
        return _RecursivePhaseEvent.apply(currents, self.threshold).flatten(0, 1)


class RecursivePhaseResNet18(ECLResNet18):
    def __init__(self, source, timesteps=4):
        _positive_timesteps(timesteps)
        super().__init__(source, timesteps=timesteps, signed=True)
        for name, node in list(self.source.named_modules()):
            if isinstance(node, DualThresholdSequence):
                parent, _, child = name.rpartition('.')
                setattr(self.source.get_submodule(parent), child,
                        RecursivePhaseSequence(node.threshold, timesteps))
        self.requires_grad_(True)
