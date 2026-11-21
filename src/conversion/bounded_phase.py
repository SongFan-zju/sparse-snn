"""Conditional shared-phase VJP for ViStream's bounded, stateful ST-BIF.

Negative threshold is exactly zero. Unlike the classification operator, the
initial membrane/count come from the preceding video frame and counts are
bounded by the source quantizer. Hard forward always uses the unperturbed state.
"""
import torch


def _vjp(x, q, c, theta, L, grad):
    # All count boundaries are processed together. Each boundary remembers the
    # current prefix that produced it; a reduction collects its contribution to
    # every earlier input. This is the same conditional Jacobian as the scalar
    # count loop, with fewer accelerator launches and no atomic accumulation.
    T, n = x.shape
    result = torch.zeros_like(x)
    levels = torch.arange(1, L+1, device=x.device, dtype=x.dtype)[:, None]
    for lo in range(0, n, 131072):
        hi = min(n, lo + 131072)
        xx, qq, cc = x[:, lo:hi], q[lo:hi], c[lo:hi]
        neg = torch.full_like(qq[None], -torch.inf)
        pos = torch.full_like(neg, torch.inf)
        initial = torch.where(cc[None] >= levels, neg, pos)
        bounds = torch.cat([neg, initial, pos], 0)
        origins = torch.full_like(bounds, -1, dtype=torch.int16)
        charge = torch.zeros_like(qq)
        contributions, sources = [], []
        for t in range(T):
            charge = charge + xx[t] / theta
            crossing = levels - cc[None] - qq[None] - charge[None]
            mid = torch.maximum(bounds[:-2], crossing)
            mid_origin = torch.where(crossing >= bounds[:-2], t, origins[:-2])
            boundary = torch.minimum(bounds[2:], mid)
            origin = torch.where(mid <= bounds[2:], mid_origin, origins[2:])
            coefficient = grad[t, lo:hi]
            if t + 1 < T:
                coefficient = coefficient - grad[t+1, lo:hi]
            contributions.append(coefficient[None] * (1-boundary.abs()).clamp_min(0))
            sources.append(origin)
            bounds = torch.cat([neg, boundary, pos], 0)
            origins = torch.cat([origins[:1], origin, origins[-1:]], 0)
        values = torch.stack(contributions)
        active_prefix = torch.stack(sources)
        for s in range(T):
            result[s, lo:hi] = (values * (active_prefix >= s)).sum((0, 1))
    return result


class _BoundedPhase(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, q0, c0, theta, L):
        q, c = q0, c0
        outputs = []
        for tick in x:
            q = q + tick / theta
            e = ((q >= 1) & (c < L)).to(q.dtype) - ((q < 0) & (c > 0)).to(q.dtype)
            q, c = q - e, c + e
            outputs.append(e * theta)
        ctx.save_for_backward(x, q0, c0, theta)
        ctx.L = L
        return torch.stack(outputs)

    @staticmethod
    def backward(ctx, grad):
        x, q, c, theta = ctx.saved_tensors
        gx = _vjp(x.flatten(1), q.flatten(), c.flatten(), theta,
                  ctx.L, grad.flatten(1))
        return gx.reshape_as(x), None, None, None, None


def bounded_phase_events(currents, q0, c0, theta, L):
    """Physical event charges with a conditional mean-Jacobian backward.

    q0 is normalized membrane, c0 is integer-valued count, theta is a fixed
    positive scalar. Initial state is detached at frame/window boundaries.
    Shared phase offset has fixed density (1-|offset|)+ on [-1,1].
    """
    return _BoundedPhase.apply(currents, q0.detach(), c0.detach(), theta, int(L))
