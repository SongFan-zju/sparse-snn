"""T2 phase-adjoint mechanism control: retain J00/J11 and remove only J10.

The deterministic signed-ECL forward and local conditional means are exactly
the retained implementation, including its boundary derivative convention.
Only the first-current VJP omits the second-output contribution. This is a
backward ablation, not a new event model or a gradient of the hard network.
"""
from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

from src.conversion.phase_moments import conditional_event_moments
from src.conversion.joint_phase_adjoint import hard_pair
from src.conversion.qcfs_ecl_resnet18 import DualThresholdSequence, ECLResNet18


class DiagonalPhaseEvent(torch.autograd.Function):
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
            mean0, mean1, *_ = conditional_event_moments(first, second, threshold, 1.)
            # Keep each diagonal block of the SAME local phase Jacobian.
            # mean0 and mean1 share the first-event boundary computation.
            first_grad, = torch.autograd.grad(
                threshold * mean0, first, grad_outputs=grad0,
                retain_graph=True, create_graph=False)
            second_grad, = torch.autograd.grad(
                threshold * mean1, second, grad_outputs=grad1,
                create_graph=False)
        return first_grad, second_grad, None


class DiagonalPhaseSequence(DualThresholdSequence):
    def __init__(self, threshold):
        super().__init__(threshold, timesteps=2, signed=True)

    def forward(self, x):
        if not torch.is_grad_enabled():
            return super().forward(x)
        first, second = x.unflatten(0, (2, -1)).unbind(0)
        return torch.stack(DiagonalPhaseEvent.apply(first, second, self.threshold)).flatten(0, 1)


class DiagonalPhaseResNet18(ECLResNet18):
    def __init__(self, source):
        super().__init__(source, timesteps=2, signed=True)
        for name, node in list(self.source.named_modules()):
            if isinstance(node, DualThresholdSequence):
                parent, _, child = name.rpartition('.')
                setattr(self.source.get_submodule(parent), child,
                        DiagonalPhaseSequence(node.threshold))
        self.requires_grad_(True)
