"""Slow phase-cell reference retained for validating the count-threshold engine.

Backward integrates initial membrane a in [-theta/2, 3*theta/2], with density
rho(a/theta)/theta and rho(z)=max(1-abs(z-1/2),0). Four threshold boundaries
per input prefix partition this interval into at most 4*T+1 cells. The actual
signed state machine is evaluated in every cell, including count eligibility.
This is O(T**2) local work, chunked over neurons; it does not use the special
uniform-[0,1] cumulative-charge recurrence.

Hard forward preserves ECL's ordered floating-point operations and theta/2
initialization. At coincident boundaries, a deterministic one-sided Jacobian
is selected by the limit of adding the same positive perturbation to all
currents: later-prefix boundaries precede earlier ones, and cell ranks resolve
the zero-width intermediate trajectories. This can differ from the retained
closed-form T2 implementation's tie convention. Neither surrogate is an exact hard
loss gradient or an unbiased whole-network stochastic-loss gradient.
"""
from __future__ import annotations

import torch

from src.conversion.general_phase_adjoint import (
    _threshold_chunk,
)
from src.conversion.triangular_density_phase import triangular_density_cdf


# An implementation memory bound, not a neuron/model hyperparameter.
_CHUNK_NEURONS = 262144


def _geometry(currents, theta):
    total = torch.zeros_like(theta)
    boundaries = []
    for current in currents:
        total = total + current
        positive = torch.remainder(-total + .5 * theta, theta) - .5 * theta
        negative = torch.remainder(-total - 1e-3 + .5 * theta, theta) - .5 * theta
        boundaries.extend((positive, positive + theta, negative, negative + theta))
    boundary_values = torch.stack(boundaries).unflatten(0, (len(currents), 4))
    # b_t(I + delta*1) = b_t(I) - (t+1)*delta. For exactly tied boundaries,
    # stable sorting in reverse time selects this fixed one-sided limit.
    boundaries, order = boundary_values.flip(0).flatten(0, 1).sort(dim=0, stable=True)
    origin = len(currents) - 1 - order // 4
    positions = torch.arange(len(boundaries), device=currents.device)[:, None].expand_as(order)
    inverse = torch.empty_like(order).scatter_(0, order, positions)
    boundary_ranks = inverse.unflatten(0, (len(currents), 4)).flip(0)
    lower, upper = -.5 * theta, 1.5 * theta
    valid = (boundaries > lower) & (boundaries < upper)
    endpoints = torch.cat((lower[None], boundaries, upper[None]), dim=0)
    centers = endpoints[:-1] + .5 * (endpoints[1:] - endpoints[:-1])
    return centers, endpoints, boundaries, origin, valid, boundary_values, boundary_ranks


def _partition_chunk(currents, theta, gradient):
    centers, endpoints, boundaries, origin, valid, boundary_values, boundary_ranks = _geometry(currents, theta)
    cell_ranks = torch.arange(len(centers), device=currents.device)[:, None]
    if gradient is None:
        cdf = triangular_density_cdf(endpoints / theta)
        charge_weights = theta * (cdf[1:] - cdf[:-1])
        means = torch.empty_like(currents)
        scores = None
    else:
        scores = torch.zeros_like(centers)
    total = torch.zeros_like(theta)
    count = torch.zeros_like(centers)
    for t, current in enumerate(currents):
        total = total + current
        # The real-arithmetic charge identity is only used inside the integral.
        # Deployment still uses hard_sequence's physical operation order.
        before_reset = centers + total
        membrane = before_reset - theta * count
        positive = membrane > theta
        negative = membrane <= -1e-3
        # A zero-width cell at coincident boundaries represents a real cell
        # after the stated infinitesimal perturbation. Floating comparisons at
        # its unperturbed center cannot distinguish those intermediate states.
        # Lattice levels identify the relevant count-dependent threshold; rank
        # comparisons then select its side without a tunable numeric epsilon.
        positive_level = (before_reset / theta).round() == count + 1
        negative_level = ((before_reset + 1e-3) / theta).round() == count
        for index in (0, 1):
            at_boundary = (centers == boundary_values[t, index]) & positive_level
            positive = torch.where(at_boundary, cell_ranks > boundary_ranks[t, index], positive)
        for index in (2, 3):
            at_boundary = (centers == boundary_values[t, index]) & negative_level
            negative = torch.where(at_boundary, cell_ranks <= boundary_ranks[t, index], negative)
        event = positive.to(currents.dtype) - (negative & (count >= 1)).to(currents.dtype)
        count.add_(event)
        if gradient is None:
            means[t] = (charge_weights * event).sum(dim=0)
        else:
            scores.add_(event * gradient[t])
    if gradient is None:
        return means
    density = (1 - (boundaries / theta - .5).abs()).clamp_min(0)
    jumps = (scores[1:] - scores[:-1]) * density
    jumps.masked_fill_(~valid, 0.)
    result = torch.empty_like(currents)
    suffix = torch.zeros_like(theta)
    for t in range(len(currents) - 1, -1, -1):
        # d(boundary)/dI_k=-1 for k<=origin. The moving-cell sign cancels
        # that minus sign; charge theta cancels the density's 1/theta.
        suffix = suffix + torch.where(origin == t, jumps, 0.).sum(dim=0)
        result[t] = suffix
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
    """Piecewise local charge-mean VJP, with threshold fixed and no graph."""
    return _partition(currents, threshold, output_gradient)


