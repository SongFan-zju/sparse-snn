"""Exact two-step conditional signed-event moments under a shared initial phase."""
import torch


def conditional_event_moments(current0, current1, threshold, phase_width=1.):
    """Return E[s0], E[s1], E[s0^2], E[s1^2], E[s0*s1], s1 in{-1,0,1}.

    a=I0/theta, b=I1/theta; p is the SAME initial membrane fraction at both
    steps. Its interval[lo,hi] is centered at1/2, with width in(0,1]. Intersect
    the second-event inequalities with p in[lo,A] or[A,hi], A=clip(1-a,lo,hi).
    Strict endpoint
    choices have measure zero for this continuous phase integral.
    """
    width = torch.as_tensor(phase_width,device=current0.device,dtype=current0.dtype)
    lower,upper = (1-width)/2,(1+width)/2
    first_boundary = torch.minimum(torch.maximum(1-current0/threshold,lower),upper)
    total = (current0+current1)/threshold
    first = (upper-first_boundary)/width
    positive_after_zero = (first_boundary-torch.maximum(lower,1-total)).clamp_min(0)/width
    positive_after_one = (upper-torch.maximum(first_boundary,2-total)).clamp_min(0)/width
    negative_after_one = (torch.minimum(upper,1-total-1e-3/threshold)-first_boundary).clamp_min(0)/width
    positive = positive_after_zero+positive_after_one
    second = positive-negative_after_one
    return first,second,first,positive+negative_after_one,positive_after_one-negative_after_one
