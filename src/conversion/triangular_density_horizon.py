"""General-horizon signed ECL with a shared triangular-phase local VJP.

Backward integrates initial membrane a in [-theta/2, 3*theta/2], with density
rho(a/theta)/theta and rho(z)=max(1-abs(z-1/2),0). A min/max recurrence tracks
the initial-membrane boundaries of cumulative count C_t >= k. Integrating
their CDFs gives the full emitted-charge mean without sorting phase cells.
This count-threshold recurrence applies to arbitrary initial-phase CDFs;
it is distinct from the special uniform-[0,1] scalar charge recurrence.

Hard forward preserves ECL's ordered floating-point operations and theta/2
initialization. At coincident boundaries, a deterministic one-sided Jacobian
is selected by the limit of adding the same positive perturbation to all
currents: max ties choose earlier origins, min ties later origins. This can differ from the retained
closed-form T2 implementation's tie convention. Neither surrogate is an exact hard
loss gradient or an unbiased whole-network stochastic-loss gradient.
"""
from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

from src.conversion.general_phase_adjoint import (
    _positive_integer, _threshold_chunk, hard_sequence,
)
from src.conversion.qcfs_ecl_resnet18 import DualThresholdSequence, ECLResNet18
from src.conversion.triangular_density_phase import triangular_density_cdf


# An implementation memory bound, not a neuron/model hyperparameter.
_CHUNK_NEURONS = 4194304


def _partition_chunk(currents, theta, gradient):
    """Count-threshold recurrence, retaining only one horizon slice of state.

    b[t,k] = min(b[t-1,k+1], max(b[t-1,k], k*theta-S[t]-epsilon),
                 max(b[t-1,k-1], k*theta-S[t])).
    b[t,0]=-inf and b[0,k>=1]=+inf. A finite selected boundary has
    derivative -1 for input times <= its origin, and zero thereafter.
    """
    boundaries = currents.new_empty((0, len(theta)))
    origins = torch.empty((0, len(theta)), dtype=torch.int32, device=currents.device)
    infinity = torch.full_like(theta[None], float('inf'))
    constant_origin = torch.full((1, len(theta)), -1, dtype=torch.int32,
                                 device=currents.device)
    total = torch.zeros_like(theta)
    previous_charge = torch.zeros_like(theta)
    result = torch.empty_like(currents) if gradient is None else torch.zeros_like(currents)
    for t, current in enumerate(currents):
        total = total + current
        padded = torch.cat((-infinity, boundaries, infinity, infinity), dim=0)
        lower, same, higher = padded[:t+1], padded[1:t+2], padded[2:t+3]
        levels = torch.arange(1, t + 2, dtype=currents.dtype,
                              device=currents.device)[:, None]
        positive = levels * theta - total
        negative = positive - 1e-3
        hold_or_retract = torch.maximum(same, negative)
        emit = torch.maximum(lower, positive)
        candidate = torch.minimum(hold_or_retract, emit)
        boundaries = torch.minimum(higher, candidate)
        if gradient is None:
            charge = theta * (1 - triangular_density_cdf(boundaries / theta)).sum(dim=0)
            result[t] = charge - previous_charge
            previous_charge = charge
        else:
            padded_origins = torch.cat((constant_origin, origins,
                                        constant_origin, constant_origin), dim=0)
            lo_origin = padded_origins[:t+1]
            same_origin = padded_origins[1:t+2]
            hi_origin = padded_origins[2:t+3]
            # Old origins precede t, so max ties select the old candidate.
            hold_origin = torch.where(same >= negative, same_origin, t)
            emit_origin = torch.where(lower >= positive, lo_origin, t)
            choose_hold = ((hold_or_retract < emit)
                           | ((hold_or_retract == emit) & (hold_origin >= emit_origin)))
            candidate_origin = torch.where(choose_hold, hold_origin, emit_origin)
            choose_higher = ((higher < candidate)
                             | ((higher == candidate) & (hi_origin >= candidate_origin)))
            origins = torch.where(choose_higher, hi_origin, candidate_origin)
            density = (1 - (boundaries / theta - .5).abs()).clamp_min(0)
            # Sum_t g[t]*mu[t] = theta*sum_t (g[t]-g[t+1])*E[C_t].
            adjoint = gradient[t] - gradient[t+1] if t+1 < len(currents) else gradient[t]
            weighted = density * adjoint
            for input_time in range(t + 1):
                result[input_time].add_(torch.where(
                    origins >= input_time, weighted, 0.).sum(dim=0))
    return result


def _partition(currents, threshold, gradient):
    if currents.ndim < 2 or currents.shape[0] < 1 or not currents.is_floating_point():
        raise ValueError('Expected floating currents shaped [T, *neuron_shape]')
    shape = tuple(currents.shape[1:])
    if threshold.ndim > len(shape) or torch.broadcast_shapes(shape, threshold.shape) != shape:
        raise ValueError('Threshold must broadcast to the neuron shape')
    if gradient is not None and (gradient.shape != currents.shape
            or gradient.device != currents.device or gradient.dtype != currents.dtype):
        raise ValueError('Output gradient must match currents')
    threshold = threshold.to(device=currents.device, dtype=currents.dtype)
    flat = currents.reshape(currents.shape[0], -1)
    weights = None if gradient is None else gradient.reshape_as(flat)
    result = torch.empty_like(flat)
    for start in range(0, flat.shape[1], _CHUNK_NEURONS):
        stop = min(start + _CHUNK_NEURONS, flat.shape[1])
        theta = _threshold_chunk(threshold, shape, start, stop)
        result[:, start:stop] = _partition_chunk(
            flat[:, start:stop], theta,
            None if weights is None else weights[:, start:stop])
    return result.reshape_as(currents)


@torch.no_grad()
def triangle_density_partition_mean(currents, threshold):
    """Conditional emitted-charge means; threshold broadcasts over neurons."""
    return _partition(currents, threshold, None)


@torch.no_grad()
def triangle_density_partition_vjp(currents, threshold, output_gradient):
    """Same local VJP; eligible CUDA T3/T4/T7/T15 use optional fused Triton.

    All other cases retain the original Torch implementation. Kernel errors
    propagate; only an unavailable optional dependency selects the fallback.
    """
    # Retain the original validation before selecting either implementation.
    if currents.ndim < 2 or currents.shape[0] < 1 or not currents.is_floating_point():
        raise ValueError('Expected floating currents shaped [T, *neuron_shape]')
    shape = tuple(currents.shape[1:])
    if threshold.ndim > len(shape) or torch.broadcast_shapes(shape, threshold.shape) != shape:
        raise ValueError('Threshold must broadcast to the neuron shape')
    if (output_gradient.shape != currents.shape or output_gradient.device != currents.device
            or output_gradient.dtype != currents.dtype):
        raise ValueError('Output gradient must match currents')
    from .triangular_density_triton import is_supported, triton_density_vjp
    if is_supported(currents, threshold):
        return triton_density_vjp(currents, threshold, output_gradient)
    return _partition(currents, threshold, output_gradient)


class _TriangularDensityHorizonEvent(torch.autograd.Function):
    @staticmethod
    def forward(ctx, currents, threshold):
        ctx.save_for_backward(currents, threshold)
        return hard_sequence(currents, threshold)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        currents, threshold = ctx.saved_tensors
        return triangle_density_partition_vjp(currents, threshold, gradient), None


class TriangularDensityHorizonSequence(DualThresholdSequence):
    def __init__(self, threshold, timesteps=4):
        _positive_integer(timesteps, 'timesteps')
        super().__init__(threshold, timesteps=timesteps, signed=True)

    def forward(self, x):
        if not torch.is_grad_enabled():
            return super().forward(x)
        currents = x.unflatten(0, (self.timesteps, -1))
        return _TriangularDensityHorizonEvent.apply(
            currents, self.threshold).flatten(0, 1)


class TriangularDensityHorizonResNet18(ECLResNet18):
    """ECL-compatible state and forward, including standard CIFAR ResNet20."""
    def __init__(self, source, timesteps=4):
        _positive_integer(timesteps, 'timesteps')
        super().__init__(source, timesteps=timesteps, signed=True)
        for name, node in list(self.source.named_modules()):
            if isinstance(node, DualThresholdSequence):
                parent, _, child = name.rpartition('.')
                setattr(self.source.get_submodule(parent), child,
                        TriangularDensityHorizonSequence(node.threshold, timesteps))
        self.requires_grad_(True)
