"""Hard ECL T2 with a shared-initial-phase conditional mean Jacobian.

The forward is the unchanged deterministic signed ECL trajectory. Only its
backward uses the two-event mean integrated over one shared initial phase,
conditional on the actual incoming currents. No moments are propagated through
convolutions or normalization. Thresholds are frozen at the converted source.

This is a biased local surrogate, not an exact gradient of the deterministic
network or an unbiased gradient of a globally stochastic loss. Generic hard
forward / approximate backward and temporal marginalization are prior art.
The concrete candidate is the coupled reset/retraction-aware phase Jacobian.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.autograd.function import once_differentiable

from src.conversion.phase_moments import conditional_event_moments
from src.conversion.qcfs_ecl_resnet18 import DualThresholdSequence, ECLResNet18


def hard_pair(current0, current1, threshold):
    membrane = .5 * threshold + current0
    event0 = (membrane > threshold).to(current0)
    output0 = threshold * event0
    membrane = membrane - output0
    membrane = membrane + current1
    event1 = ((membrane > threshold).to(current1)
              - ((membrane <= -1e-3) & (event0 >= 1)).to(current1))
    return output0, threshold * event1


class JointPhaseEvent(torch.autograd.Function):
    @staticmethod
    def forward(ctx, current0, current1, threshold):
        ctx.save_for_backward(current0, current1, threshold)
        return hard_pair(current0, current1, threshold)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad0, grad1):
        current0, current1, threshold = ctx.saved_tensors
        # Recompute only the local conditional means, avoiding a saved
        # cross-layer moment graph. One full phase interval is fixed in advance.
        with torch.enable_grad():
            first = current0.detach().requires_grad_(True)
            second = current1.detach().requires_grad_(True)
            mean0, mean1, *_ = conditional_event_moments(first, second, threshold, 1.)
            gradients = torch.autograd.grad(
                (threshold * mean0, threshold * mean1), (first, second),
                grad_outputs=(grad0, grad1), create_graph=False)
        return gradients[0], gradients[1], None


class JointPhaseSequence(DualThresholdSequence):
    def __init__(self, threshold):
        super().__init__(threshold, timesteps=2, signed=True)

    def forward(self, x):
        if not torch.is_grad_enabled():
            return super().forward(x)
        first, second = x.unflatten(0, (2, -1)).unbind(0)
        return torch.stack(JointPhaseEvent.apply(first, second, self.threshold)).flatten(0, 1)


class JointPhaseResNet18(ECLResNet18):
    def __init__(self, source):
        super().__init__(source, timesteps=2, signed=True)
        for name, node in list(self.source.named_modules()):
            if isinstance(node, DualThresholdSequence):
                parent, _, child = name.rpartition('.')
                setattr(self.source.get_submodule(parent), child,
                        JointPhaseSequence(node.threshold))
        self.requires_grad_(True)

    def forward(self, images):
        # Preserve ECL's operation order and its single final classifier call.
        return super().forward(images)
