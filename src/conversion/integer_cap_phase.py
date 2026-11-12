"""Signed integer-capacity events with the existing local phase-mean VJP.

This is a known event-precision extension, not binary single-event ECL.
For M=3 the magnitude 0..3 has two fixed dyadic bits (weights 1 and 2),
with a shared sign; this raises event capacity even when physical charge
per step stays at the original QCFS threshold. No second network pass is used.

The stored ``threshold`` remains the source QCFS threshold for strict state
mapping. The operational unit is threshold/M, initial membrane is unit/2,
and negative retraction is bounded by the previous cumulative integer count.
For phase uniform on [0, unit), the conditional mean emitted charge follows
the retained recursion with cap=M*unit. Its VJP is a local surrogate, not an
exact gradient of deterministic network loss. The T2 training path uses the
retained closed-T2 formula and its boundary convention with original theta,
without a divide/multiply round trip. Generic helpers retain recursive ties.
The general-horizon training path likewise uses the stored physical theta
directly in the retained recursion, including its boundary convention.
"""
from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

from src.conversion.phase_moments import conditional_event_moments
from src.conversion.qcfs_ecl_resnet18 import DualThresholdSequence, ECLResNet18
from src.conversion.recursive_phase_adjoint import (
    EPSILON, _coefficient_vjp, _positive_timesteps, _validate_inputs,
    recursive_phase_coefficients, recursive_phase_mean, recursive_phase_vjp,
)


def _positive_cap(max_count):
    if not isinstance(max_count, int) or isinstance(max_count, bool) or max_count < 1:
        raise ValueError('max_count must be a positive integer')


@torch.no_grad()
def integer_cap_hard_sequence(currents, unit, max_count=3):
    """One hard trajectory, emitted charges shaped [T, *neuron_shape].

    Explicit comparisons preserve signed ECL's strict positive and inclusive
    negative boundaries, including v=-epsilon. All comparisons use the same
    pre-reset membrane; reset subtracts the single total emitted charge.
    """
    _positive_cap(max_count)
    unit = _validate_inputs(currents, unit)
    membrane = .5 * unit
    count = torch.zeros_like(currents[0])
    outputs = []
    for current in currents:
        membrane = membrane + current
        positive = (membrane > unit).to(current.dtype)
        negative = (membrane <= -EPSILON).to(current.dtype)
        for j in range(2, max_count + 1):
            positive = positive + (membrane > j * unit).to(current.dtype)
            negative = negative + (membrane <= -EPSILON - (j - 1) * unit).to(current.dtype)
        event = positive - torch.minimum(negative, count)
        output = event * unit
        membrane = membrane - output
        count = count + event
        outputs.append(output)
    return torch.stack(outputs)


@torch.no_grad()
def integer_cap_phase_mean(currents, unit, max_count=3):
    """Conditional phase means with per-step physical cap M*unit."""
    _positive_cap(max_count)
    return recursive_phase_mean(currents, max_count * unit)


@torch.no_grad()
def integer_cap_phase_vjp(currents, unit, output_gradient, max_count=3):
    """Conditional-mean VJP with fixed unit and fixed integer cap."""
    _positive_cap(max_count)
    return recursive_phase_vjp(currents, max_count * unit, output_gradient)


class _IntegerCapPhaseEvent(torch.autograd.Function):
    @staticmethod
    def forward(ctx, currents, unit, max_count):
        a, b = recursive_phase_coefficients(currents, max_count * unit)
        ctx.save_for_backward(a, b)
        return integer_cap_hard_sequence(currents, unit, max_count)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        a, b = ctx.saved_tensors
        return _coefficient_vjp(a, b, gradient), None, None


class _IntegerCapPhysicalPhaseEvent(torch.autograd.Function):
    """General-horizon VJP at original theta, without unit reconstruction."""
    @staticmethod
    def forward(ctx, currents, threshold, max_count):
        a, b = recursive_phase_coefficients(currents, threshold)
        ctx.save_for_backward(a, b)
        return integer_cap_hard_sequence(currents, threshold / max_count, max_count)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        a, b = ctx.saved_tensors
        return _coefficient_vjp(a, b, gradient), None, None


class _IntegerCapT2PhaseEvent(torch.autograd.Function):
    """Retained JointPhaseEvent backward, changing only hard event precision."""
    @staticmethod
    def forward(ctx, current0, current1, threshold, max_count):
        ctx.save_for_backward(current0, current1, threshold)
        return integer_cap_hard_sequence(
            torch.stack((current0, current1)), threshold / max_count, max_count).unbind(0)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad0, grad1):
        current0, current1, threshold = ctx.saved_tensors
        # Exactly the retained JointPhaseEvent backward and boundary rules.
        # Fixed physical cap gives the same conditional means for every M.
        with torch.enable_grad():
            first = current0.detach().requires_grad_(True)
            second = current1.detach().requires_grad_(True)
            mean0, mean1, *_ = conditional_event_moments(first, second, threshold, 1.)
            gradients = torch.autograd.grad(
                (threshold * mean0, threshold * mean1), (first, second),
                grad_outputs=(grad0, grad1), create_graph=False)
        return gradients[0], gradients[1], None, None


class IntegerCapSequence(DualThresholdSequence):
    """Deployment node; threshold is the original QCFS physical charge cap."""
    def __init__(self, threshold, timesteps=2, max_count=3):
        _positive_timesteps(timesteps)
        _positive_cap(max_count)
        super().__init__(threshold, timesteps=timesteps, signed=True)
        self.max_count = max_count

    @property
    def unit(self):
        return self.threshold / self.max_count

    def forward(self, x):
        currents = x.unflatten(0, (self.timesteps, -1))
        return integer_cap_hard_sequence(currents, self.unit, self.max_count).flatten(0, 1)


class IntegerCapPhaseSequence(IntegerCapSequence):
    def forward(self, x):
        if not torch.is_grad_enabled():
            return super().forward(x)
        currents = x.unflatten(0, (self.timesteps, -1))
        if self.timesteps == 2:
            first, second = currents.unbind(0)
            return torch.stack(_IntegerCapT2PhaseEvent.apply(
                first, second, self.threshold, self.max_count)).flatten(0, 1)
        return _IntegerCapPhysicalPhaseEvent.apply(
            currents, self.threshold, self.max_count).flatten(0, 1)


class IntegerCapResNet18(ECLResNet18):
    """Single integer-event trajectory and the inherited single classifier."""
    node_type = IntegerCapSequence

    def __init__(self, source, timesteps=2, max_count=3):
        _positive_timesteps(timesteps)
        _positive_cap(max_count)
        super().__init__(source, timesteps=timesteps, signed=True)
        self.max_count = max_count
        for name, node in list(self.source.named_modules()):
            if isinstance(node, DualThresholdSequence):
                parent, _, child = name.rpartition('.')
                setattr(self.source.get_submodule(parent), child,
                        self.node_type(node.threshold, timesteps, max_count))


class IntegerCapPhaseResNet18(IntegerCapResNet18):
    node_type = IntegerCapPhaseSequence

    def __init__(self, source, timesteps=2, max_count=3):
        super().__init__(source, timesteps=timesteps, max_count=max_count)
        self.requires_grad_(True)
