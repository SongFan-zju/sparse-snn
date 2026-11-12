"""General-horizon ECL with an exact local uniform-phase mean VJP.

The hard path retains the physical operation order of DualThresholdSequence.
Only backward integrates over an initial phase in [0, theta], conditional on
fixed incoming currents and a fixed threshold. This is not an unbiased whole
network gradient. There are at most 2*T+1 phase cells; direct evaluation costs
O(T**2) per neuron. Cell storage is limited to a chunk of independent neurons.

At coincident boundaries or modulo endpoints the Jacobian need not be unique.
We zero the jump contribution of every exactly coincident or endpoint boundary.
This convention is not asserted to equal the old T2 derivative at those points.
"""
from __future__ import annotations

import math

import torch
from torch.autograd.function import once_differentiable

from src.conversion.ecl_triangular_surrogate import TriangularSequence
from src.conversion.qcfs_ecl_resnet18 import DualThresholdSequence, ECLResNet18


DEFAULT_CHUNK_NEURONS = 1048576


def _positive_integer(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f'{name} must be a positive integer')


@torch.no_grad()
def hard_sequence(currents, threshold):
    """Hard emitted charges for currents shaped [T, *neuron_shape]."""
    if currents.ndim < 2 or currents.shape[0] < 1:
        raise ValueError('Expected currents shaped [T, *neuron_shape]')
    membrane = .5 * threshold
    count = torch.zeros_like(currents[0])
    outputs = []
    for current in currents:
        membrane = membrane + current
        event = (membrane > threshold).to(current.dtype)
        event = event - ((membrane <= -1e-3) & (count >= 1)).to(current.dtype)
        output = event * threshold
        membrane = membrane - output
        count = count + event
        outputs.append(output)
    return torch.stack(outputs)


def _threshold_chunk(threshold, shape, start, stop):
    """Slice a broadcast threshold without materializing its full NCHW copy."""
    if threshold.numel() == 1:
        return threshold.reshape(1).expand(stop - start)
    padded = (1,) * (len(shape) - threshold.ndim) + tuple(threshold.shape)
    positions = torch.arange(start, stop, device=threshold.device)
    indices = torch.zeros_like(positions)
    feature_stride, threshold_stride = math.prod(shape), threshold.numel()
    for size, threshold_size in zip(shape, padded):
        feature_stride //= size
        threshold_stride //= threshold_size
        if threshold_size != 1:
            indices.add_((positions // feature_stride).remainder(size) * threshold_stride)
    return threshold.reshape(-1)[indices]


def _geometry(currents, theta):
    """Return cell centers/widths and sorted boundary times/validity."""
    total = torch.zeros_like(theta)
    boundaries = []
    for current in currents:
        # Explicit prefix loop: no CUDA cumsum under deterministic=True.
        total = total + current
        boundaries.extend((torch.remainder(-total, theta),
                           torch.remainder(-total - 1e-3, theta)))
    sorted_boundaries, order = torch.stack(boundaries).sort(dim=0, stable=True)
    origin = order // 2
    valid = (sorted_boundaries > 0) & (sorted_boundaries < theta)
    equal = sorted_boundaries[1:] == sorted_boundaries[:-1]
    valid[1:] &= ~equal
    valid[:-1] &= ~equal
    endpoints = torch.cat((torch.zeros_like(theta)[None], sorted_boundaries,
                           theta[None]), dim=0)
    widths = endpoints[1:] - endpoints[:-1]
    centers = endpoints[:-1] + .5 * widths
    return centers, widths, origin, valid


def _partition_chunk(currents, theta, gradient=None):
    centers, widths, origin, valid = _geometry(currents, theta)
    count = torch.zeros_like(centers)
    total = torch.zeros_like(theta)
    # Keep one cell score, not a T x cells x neurons event tensor or graph.
    scores = torch.zeros_like(centers) if gradient is not None else None
    means = torch.empty_like(currents) if gradient is None else None
    for t, current in enumerate(currents):
        total = total + current
        # This auxiliary integral evaluates the real charge identity. It is
        # not used to replace the deployment's ordered floating-point resets.
        membrane = centers + total - theta * count
        event = (membrane > theta).to(currents.dtype)
        event = event - ((membrane <= -1e-3) & (count >= 1)).to(currents.dtype)
        count.add_(event)
        if gradient is None:
            # E[theta*s] = sum(cell_width*s): density1/theta cancels charge theta.
            means[t] = (widths * event).sum(dim=0)
        else:
            scores.add_(event * gradient[t])
    if gradient is None:
        return means
    jumps = scores[1:] - scores[:-1]
    jumps.masked_fill_(~valid, 0.)
    result = torch.empty_like(currents)
    suffix = torch.zeros_like(theta)
    for t in range(len(currents) - 1, -1, -1):
        # Boundary derivative is -1 for every current at/before its origin.
        # Use fixed-source masks instead of scatter_add, then an explicit suffix.
        suffix = suffix + torch.where(origin == t, jumps, 0.).sum(dim=0)
        result[t] = suffix
    return result


def _partition(currents, threshold, gradient, chunk_neurons):
    _positive_integer(chunk_neurons, 'chunk_neurons')
    if currents.ndim < 2 or currents.shape[0] < 1 or not currents.is_floating_point():
        raise ValueError('Expected floating currents shaped [T, *neuron_shape]')
    shape = tuple(currents.shape[1:])
    if threshold.ndim > len(shape) or torch.broadcast_shapes(shape, threshold.shape) != shape:
        raise ValueError('Threshold must broadcast to the neuron shape')
    if gradient is not None and (gradient.shape != currents.shape or gradient.device != currents.device):
        raise ValueError('Output gradient must match currents')
    # Thresholds are already validated by the sequence constructor. Avoid a
    # synchronizing finite/positivity check on every backward GPU invocation.
    threshold = threshold.to(device=currents.device, dtype=currents.dtype)
    flat = currents.reshape(currents.shape[0], -1)
    weights = None if gradient is None else gradient.reshape_as(flat)
    result = torch.empty_like(flat)
    for start in range(0, flat.shape[1], chunk_neurons):
        stop = min(start + chunk_neurons, flat.shape[1])
        theta = _threshold_chunk(threshold, shape, start, stop)
        result[:, start:stop] = _partition_chunk(
            flat[:, start:stop], theta,
            None if weights is None else weights[:, start:stop])
    return result.reshape_as(currents)


@torch.no_grad()
def phase_partition_mean(currents, threshold, chunk_neurons=DEFAULT_CHUNK_NEURONS):
    """Conditional emitted-charge means; scalar/broadcast thresholds supported.

    This pure diagnostic function has no autograd graph. Use
    phase_partition_vjp for its stated piecewise local input derivative.
    """
    return _partition(currents, threshold, None, chunk_neurons)


@torch.no_grad()
def phase_partition_vjp(currents, threshold, output_gradient,
                        chunk_neurons=DEFAULT_CHUNK_NEURONS):
    """VJP of conditional charge means w.r.t. currents, with theta fixed."""
    return _partition(currents, threshold, output_gradient, chunk_neurons)


class _PhasePartitionEvent(torch.autograd.Function):
    @staticmethod
    def forward(ctx, currents, threshold, chunk_neurons):
        ctx.save_for_backward(currents, threshold)
        ctx.chunk_neurons = chunk_neurons
        return hard_sequence(currents, threshold)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        currents, threshold = ctx.saved_tensors
        return phase_partition_vjp(currents, threshold, gradient,
                                   ctx.chunk_neurons), None, None


class PhasePartitionSequence(DualThresholdSequence):
    def __init__(self, threshold, timesteps=4, chunk_neurons=DEFAULT_CHUNK_NEURONS):
        _positive_integer(timesteps, 'timesteps')
        _positive_integer(chunk_neurons, 'chunk_neurons')
        super().__init__(threshold, timesteps=timesteps, signed=True)
        self.chunk_neurons = chunk_neurons

    def forward(self, x):
        if not torch.is_grad_enabled():
            return super().forward(x)
        currents = x.unflatten(0, (self.timesteps, -1))
        return _PhasePartitionEvent.apply(currents, self.threshold,
                                          self.chunk_neurons).flatten(0, 1)


class TriangularHorizonSequence(TriangularSequence):
    def __init__(self, threshold, timesteps=4):
        _positive_integer(timesteps, 'timesteps')
        super().__init__(threshold)
        self.timesteps = timesteps


class PhasePartitionResNet18(ECLResNet18):
    def __init__(self, source, timesteps=4, chunk_neurons=DEFAULT_CHUNK_NEURONS):
        _positive_integer(timesteps, 'timesteps')
        _positive_integer(chunk_neurons, 'chunk_neurons')
        super().__init__(source, timesteps=timesteps, signed=True)
        self.chunk_neurons = chunk_neurons
        for name, node in list(self.source.named_modules()):
            if isinstance(node, DualThresholdSequence):
                parent, _, child = name.rpartition('.')
                setattr(self.source.get_submodule(parent), child,
                        PhasePartitionSequence(node.threshold, timesteps, chunk_neurons))
        self.requires_grad_(True)


class TriangularHorizonResNet18(ECLResNet18):
    def __init__(self, source, timesteps=4):
        _positive_integer(timesteps, 'timesteps')
        super().__init__(source, timesteps=timesteps, signed=True)
        for name, node in list(self.source.named_modules()):
            if isinstance(node, DualThresholdSequence):
                parent, _, child = name.rpartition('.')
                setattr(self.source.get_submodule(parent), child,
                        TriangularHorizonSequence(node.threshold, timesteps))
        self.requires_grad_(True)
