"""Trainable fixed-T ViStream/ST-BIF backbone, with cross-frame state.

This branch uses the author's tracking ResNet50 checkpoint. A frame is
encoded once as a delta followed by zero currents.
The time axis is folded only through linear layers; every neuron retains its
bounded event count and membrane across frames.  Video boundaries call reset().
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class QuantizedActivation(nn.Module):
    """State-dict compatible with the upstream non-symmetric MyQuan."""
    def __init__(self, L):
        super().__init__()
        self.L = int(L)
        self.s = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return (x / self.s + .5).floor().clamp(0, self.L) * self.s


def load_source(upstream, checkpoint, L):
    """Strict-load the full author QANN before removing the tracking tail."""
    path = Path(upstream) / 'model/resnet.py'
    spec = importlib.util.spec_from_file_location('_vistream_resnet_native', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.ResNet(module.Bottleneck, [3, 4, 6, 3])

    def replace(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.ReLU):
                setattr(parent, name, QuantizedActivation(L))
            else:
                replace(child)
    replace(model)
    raw = torch.load(checkpoint, map_location='cpu', weights_only=False)
    args = raw.get('args')
    if args is not None and int(args.level) // 2 - 1 != L:
        raise ValueError('L must match the count bound in the source checkpoint')
    if args is not None and int(args.weight_quantization_bit) != 32:
        raise ValueError('This loader expects the public full-precision weights')
    model.load_state_dict(raw['model'], strict=True)
    model.modify(remove_layers=['layer4'])
    return model.eval(), raw.get('args')


def hard_events(currents, q0, c0, theta, L):
    """Upstream >=1 / <0 bounded, signed, one-event circuit."""
    q, c = q0, c0
    out = []
    for current in currents:
        q = q + current / theta
        e = ((q >= 1) & (c < L)).to(q.dtype) - ((q < 0) & (c > 0)).to(q.dtype)
        q = q - e
        c = c + e
        out.append(e * theta)
    return torch.stack(out), q, c


class _TriangleEvent(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, count, L):
        ctx.save_for_backward(q, count)
        ctx.L = L
        return ((q >= 1) & (count < L)).to(q.dtype) - ((q < 0) & (count > 0)).to(q.dtype)

    @staticmethod
    def backward(ctx, grad):
        q, c = ctx.saved_tensors
        density = (1 - (q - 1).abs()).clamp_min(0) * (c < ctx.L)
        density = density + (1 - q.abs()).clamp_min(0) * (c > 0)
        return grad * density, None, None


class StatefulNeuron(nn.Module):
    def __init__(self, activation, method):
        super().__init__()
        self.register_buffer('threshold', activation.s.detach().clone())
        if not bool(torch.isfinite(self.threshold) & (self.threshold > 0)):
            raise ValueError('Invalid source quantization step')
        self.L, self.method = activation.L, method
        self.reset()

    def reset(self):
        self.q = self.c = None

    def forward(self, x):
        q = torch.full_like(x[0], .5) if self.q is None else self.q
        c = torch.zeros_like(x[0]) if self.c is None else self.c
        if not torch.is_grad_enabled() or not x.requires_grad or self.method == 'hard':
            y, q1, c1 = hard_events(x, q, c, self.threshold, self.L)
        elif self.method == 'triangular':
            events = []
            for current in x:
                q = q + current / self.threshold
                event = _TriangleEvent.apply(q, c, self.L)
                q = q - event
                c = c + event.detach()
                events.append(event * self.threshold)
            y, q1, c1 = torch.stack(events), q, c
        elif self.method == 'density':
            from .conversion.bounded_phase import bounded_phase_events
            y = bounded_phase_events(x, q, c, self.threshold, self.L)
            with torch.no_grad():
                _, q1, c1 = hard_events(x, q, c, self.threshold, self.L)
        else:
            raise ValueError(self.method)
        # Truncated BPTT: retain actual hard state without a stale prior-frame graph.
        self.q, self.c = q1.detach(), c1.detach()
        return y


class TemporalConv(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.conv = source
        self.first = True
        self.frame_ticks = None

    def reset(self):
        self.first = True

    def forward(self, x):
        t, b = x.shape[:2]
        def convolution(values):
            return F.conv2d(values.flatten(0,1), self.conv.weight, None, self.conv.stride,
                            self.conv.padding, self.conv.dilation, self.conv.groups)
        if self.frame_ticks is None or t<=self.frame_ticks:
            y = convolution(x)
        else:
            # Use the same convolution batch shape as streaming inference.
            # The surrounding temporal/autograd graph is still continuous.
            y = torch.cat([convolution(x[first:first+self.frame_ticks])
                           for first in range(0,t,self.frame_ticks)],dim=0)
        y = y.reshape(t, b, *y.shape[1:])
        if self.first and self.conv.bias is not None:
            impulse = torch.zeros_like(y)
            impulse[0] = self.conv.bias[None, :, None, None]
            y = y + impulse
        self.first = False
        return y


class TemporalBatchNorm(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.bn = source
        self.first = True

    def reset(self):
        self.first = True

    def forward(self, x):
        bn = self.bn
        scale = (bn.weight / (bn.running_var + bn.eps).sqrt()).view(1, -1, 1, 1)
        y = x * scale
        if self.first:
            # Match author's arithmetic on the very first tick; later only linear.
            first = ((x[0] - bn.running_mean[None, :, None, None]) /
                     (bn.running_var + bn.eps).sqrt()[None, :, None, None])
            first = first * bn.weight[None, :, None, None] + bn.bias[None, :, None, None]
            y = torch.cat([first.unsqueeze(0), y[1:]], dim=0)
        self.first = False
        return y


class TemporalMaxPool(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.pool = source
        self.reset()

    def reset(self):
        self.accumulation = None

    def forward(self, x):
        acc = self.accumulation
        ys = []
        for tick in x:
            previous = None if acc is None else self.pool(acc)
            acc = tick if acc is None else acc + tick
            pooled = self.pool(acc)
            ys.append(pooled if previous is None else pooled - previous)
        self.accumulation = acc.detach()
        return torch.stack(ys)


class ViStreamBackbone(nn.Module):
    def __init__(self, source, T, method='density', train_last_stage=True):
        super().__init__()
        if int(T) < 1:
            raise ValueError('T must be positive')
        self.T, self.method = int(T), method
        self.source = source
        def replace(parent):
            for name, child in list(parent.named_children()):
                if isinstance(child, QuantizedActivation):
                    child = StatefulNeuron(child, method)
                elif isinstance(child, nn.Conv2d):
                    child = TemporalConv(child)
                elif isinstance(child, nn.BatchNorm2d):
                    child = TemporalBatchNorm(child)
                elif isinstance(child, nn.MaxPool2d):
                    child = TemporalMaxPool(child)
                else:
                    replace(child)
                setattr(parent, name, child)
        replace(source)
        if train_last_stage:
            for name, p in self.source.named_parameters():
                p.requires_grad_(name.startswith('layer3.'))
        self.reset()

    def reset(self):
        self.last_input = self.output_state = None
        for m in self.source.modules():
            if isinstance(m, (StatefulNeuron, TemporalConv, TemporalBatchNorm, TemporalMaxPool)):
                m.reset()

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError('Expected one current frame per batch, NCHW')
        delta = x if self.last_input is None else x - self.last_input
        ticks = torch.cat([delta.unsqueeze(0), torch.zeros_like(delta).unsqueeze(0).expand(self.T-1, *delta.shape)])
        # Upstream Bottleneck accepts temporal leading dimensions except its
        # Conv/BN/ReLU modules, which have been replaced above.
        y = self.source(ticks).sum(0)
        out = y if self.output_state is None else self.output_state + y
        self.output_state, self.last_input = out.detach(), x.detach()
        return out
