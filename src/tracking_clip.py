"""Clip-consistent training view; ordinary streaming inference is inherited."""
from __future__ import annotations

import torch

from src.tracking_vistream import ViStreamBackbone, TemporalConv


CONTEXT_RUNTIME = 'fp32_native_frame_convolution_native_cuda_roi_v1'


def configure_context_runtime(device, seed):
    from src.runtime import configure
    device = configure(device,seed,4)
    # Keep deterministic FP32 convolutions. The author's CUDA ROIAlign uses
    # atomic scatter in backward; permit that native operator rather than the
    # much larger compiled deterministic ROIAlign decomposition.
    torch.use_deterministic_algorithms(False)
    return device


class ClipViStreamBackbone(ViStreamBackbone):
    """Use one parameter/state trajectory and one neuron VJP for a short clip.

    Training chooses four frames per clip (the final clip may be shorter).
    This does not change the neuron's per-frame T or the inference interface.
    The trainer must finish backward/step before starting another clip.
    """

    def forward_clip(self, frames):
        if frames.ndim != 4 or frames.shape[0] < 1:
            raise ValueError('Expected nonempty clip [frames,C,H,W]')
        self.reset()
        ticks = []
        for index in range(len(frames)):
            delta = frames[index:index+1]
            if index:
                delta = delta - frames[index-1:index]
            first = delta.unsqueeze(0)
            zeros = torch.zeros_like(first).expand(self.T-1, *first.shape[1:])
            ticks.append(torch.cat([first, zeros], dim=0))
        emissions = self.source(torch.cat(ticks, dim=0))
        outputs, accumulated = [], None
        # Keep the same per-frame sum and ordered additions as streaming.
        # No CUDA cumsum, inter-frame detach, or additional backbone pass.
        for index in range(len(frames)):
            charge = emissions[index*self.T:(index+1)*self.T].sum(0)
            accumulated = charge if accumulated is None else accumulated + charge
            outputs.append(accumulated)
        self.output_state = accumulated.detach()
        self.last_input = frames[-1:].detach()
        return outputs

    def forward_clip_with_context(self, context, frames):
        """Rebuild current-weight history and target states in one full graph.

        Only target outputs are supervised. Keeping the prefix on this graph
        preserves the source Conv/BN bias impulse paths, which would otherwise
        be lost by a no-grad streaming warmup before the target window.
        Each call resets and recomputes history, so no old-weight state is used.
        """
        if frames.ndim != 4 or context.ndim != 4:
            raise ValueError('Expected context and target NCHW frame sequences')
        if context.shape[1:] != frames.shape[1:]:
            raise ValueError('Context and target frame dimensions differ')
        if not len(frames):
            raise ValueError('Target clip must not be empty')
        convolutions = [(module,module.frame_ticks) for module in self.source.modules()
                        if isinstance(module,TemporalConv)]
        try:
            for module,_ in convolutions:
                module.frame_ticks = self.T
            outputs = self.forward_clip(torch.cat([context, frames],dim=0))
            return outputs[len(context):]
        finally:
            for module,previous in convolutions:
                module.frame_ticks = previous
