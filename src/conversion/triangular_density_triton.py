"""Optional equivalent full-boundary Triton VJP kernels for T3/T4/T7/T15.

Validated with optional triton==3.1.0 on Torch 2.5.1 / CUDA 11.8. This module
never imports the public VJP: the caller retains the original Torch fallback.
Generated modules are cached outside src and frozen snapshots. Actual JIT or
kernel execution errors propagate; only missing optional Triton is bypassed.
"""
from __future__ import annotations
import hashlib
import os
import importlib.util
from pathlib import Path
import sys

import torch

_KERNELS = {}
_GENERATOR_VERSION = 2


def _generated_unrolled_source(steps):
    if steps not in (3, 4, 7, 15):
        raise ValueError('Only T3, T4, T7 and T15 are generated')
    lines = ['import triton', 'import triton.language as tl', '', '@triton.jit',
             'def kernel(X, TH, G, Y, N, BLOCK: tl.constexpr):',
             '    j = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)',
             '    active = j < N', '    theta = tl.load(TH)',
             '    prefix = tl.full((BLOCK,), 0., tl.float32)']
    for s in range(steps):
        lines.append(f'    out_{s} = tl.full((BLOCK,), 0., tl.float32)')
        lines.append(f'    g_{s} = tl.load(G + {s} * N + j, active, other=0.)')
    for t in range(steps):
        lines.append(f'    prefix = prefix + tl.load(X + {t} * N + j, active, other=0.)')
        lines.append(f'    adj_{t} = g_{t} - g_{t+1}' if t+1 < steps else f'    adj_{t} = g_{t}')
        for k in range(1,t+2):
            def prev(rank, origin=False):
                if rank == 0:
                    return '-1' if origin else '-float("inf")'
                if rank > t:
                    return '-1' if origin else 'float("inf")'
                return f'{"o" if origin else "b"}_{t-1}_{rank}'
            stem=f'v_{t}_{k}'
            lines.extend([
                f'    {stem}_p = {float(k)!r} * theta - prefix',
                f'    {stem}_n = {stem}_p - 0.001',
                f'    {stem}_hold = tl.maximum({prev(k)}, {stem}_n)',
                f'    {stem}_emit = tl.maximum({prev(k-1)}, {stem}_p)',
                f'    {stem}_candidate = tl.minimum({stem}_hold, {stem}_emit)',
                f'    b_{t}_{k} = tl.minimum({prev(k+1)}, {stem}_candidate)',
                f'    {stem}_ho = tl.where({prev(k)} >= {stem}_n, {prev(k,True)}, {t})',
                f'    {stem}_eo = tl.where({prev(k-1)} >= {stem}_p, {prev(k-1,True)}, {t})',
                f'    {stem}_choose_hold = ({stem}_hold < {stem}_emit) | (({stem}_hold == {stem}_emit) & ({stem}_ho >= {stem}_eo))',
                f'    {stem}_co = tl.where({stem}_choose_hold, {stem}_ho, {stem}_eo)',
                f'    {stem}_choose_hi = ({prev(k+1)} < {stem}_candidate) | (({prev(k+1)} == {stem}_candidate) & ({prev(k+1,True)} >= {stem}_co))',
                f'    o_{t}_{k} = tl.where({stem}_choose_hi, {prev(k+1,True)}, {stem}_co)',
                f'    w_{t}_{k} = tl.maximum(1. - tl.abs(tl.div_rn(b_{t}_{k}, theta) - 0.5), 0.) * adj_{t}',
            ])
        for s in range(t+1):
            # Keep the old outer-time accumulation: sum rank contributions,
            # then add this prefix's sum to its input-time gradient.
            terms=[f'tl.where(o_{t}_{k} >= {s}, w_{t}_{k}, 0.)' for k in range(1,t+2)]
            lines.append(f'    out_{s} = out_{s} + (' + ' + '.join(terms) + ')')
    for s in range(steps):
        lines.append(f'    tl.store(Y + {s} * N + j, out_{s}, active)')
    return '\n'.join(lines)+'\n'


def _generated_loop_source(steps):
    # A carried-state time loop prevents a large SSA kernel's late compiler
    # scheduler from keeping every prefix live and spilling to local memory.
    # All ranks are retained, including unreachable +inf ranks; no mobile
    # boundary reduction, clipped approximation, or derivative change occurs.
    lines=['import triton', 'import triton.language as tl', '', '@triton.jit',
           'def kernel(X, TH, G, Y, N, BLOCK: tl.constexpr):',
           '    j = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)',
           '    active = j < N', '    theta = tl.load(TH)',
           '    prefix = tl.full((BLOCK,), 0., tl.float32)']
    for k in range(1,steps+1):
        lines.extend([f'    b_{k} = tl.full((BLOCK,), float("inf"), tl.float32)',
                      f'    o_{k} = tl.full((BLOCK,), -1, tl.int32)'])
    for i in range(steps):
        lines.append(f'    out_{i} = tl.full((BLOCK,), 0., tl.float32)')
    lines.extend([f'    for t in tl.range(0, {steps}, num_stages=1):',
                  '        prefix = prefix + tl.load(X + t * N + j, active, other=0.)',
                  '        g = tl.load(G + t * N + j, active, other=0.)',
                  f'        next_g = tl.load(G + (t+1) * N + j, active & (t+1 < {steps}), other=0.)',
                  '        adj = g - next_g'])
    for k in range(1,steps+1):
        lo='-float("inf")' if k==1 else f'b_{k-1}'
        low_o='-1' if k==1 else f'o_{k-1}'
        hi='float("inf")' if k==steps else f'b_{k+1}'
        hi_o='-1' if k==steps else f'o_{k+1}'
        lines.extend([
            f'        p_{k} = {float(k)!r} * theta - prefix',
            f'        n_{k} = p_{k} - 0.001',
            f'        hold_{k} = tl.maximum(b_{k}, n_{k})',
            f'        emit_{k} = tl.maximum({lo}, p_{k})',
            f'        candidate_{k} = tl.minimum(hold_{k}, emit_{k})',
            f'        nb_{k} = tl.minimum({hi}, candidate_{k})',
            f'        ho_{k} = tl.where(b_{k} >= n_{k}, o_{k}, t)',
            f'        eo_{k} = tl.where({lo} >= p_{k}, {low_o}, t)',
            f'        choose_hold_{k} = (hold_{k} < emit_{k}) | ((hold_{k} == emit_{k}) & (ho_{k} >= eo_{k}))',
            f'        co_{k} = tl.where(choose_hold_{k}, ho_{k}, eo_{k})',
            f'        choose_high_{k} = ({hi} < candidate_{k}) | (({hi} == candidate_{k}) & ({hi_o} >= co_{k}))',
            f'        no_{k} = tl.where(choose_high_{k}, {hi_o}, co_{k})',
            f'        w_{k} = tl.maximum(1. - tl.abs(tl.div_rn(nb_{k}, theta) - 0.5), 0.) * adj'])
    for i in range(steps):
        terms=[f'tl.where(no_{k} >= {i}, w_{k}, 0.)' for k in range(1,steps+1)]
        lines.append(f'        out_{i} = out_{i} + ('+' + '.join(terms)+')')
    for k in range(1,steps+1):
        lines.extend([f'        b_{k} = nb_{k}',f'        o_{k} = no_{k}'])
    for i in range(steps):
        lines.append(f'    tl.store(Y + {i} * N + j, out_{i}, active)')
    return '\n'.join(lines)+'\n'


def generated_source(steps):
    if steps not in (3,4,7,15):
        raise ValueError('Only T3, T4, T7 and T15 are generated')
    return _generated_loop_source(steps) if steps==15 else _generated_unrolled_source(steps)


def _kernel(steps):
    if steps in _KERNELS:
        return _KERNELS[steps]
    import triton
    source=generated_source(steps)
    fingerprint=hashlib.sha256(source.encode()).hexdigest()[:16]
    name=f'_detection_density_t{steps}_v{_GENERATOR_VERSION}_{fingerprint}'
    default_cache=Path(os.environ.get('XDG_CACHE_HOME',Path.home()/'.cache'))/'sparse_snn'/'triton_density_sources'
    destination=Path(os.environ.get('SPARSE_SNN_TRITON_SOURCE_CACHE',default_cache))
    destination.mkdir(parents=True,exist_ok=True)
    path=destination/(name+'.py')
    if not path.exists() or path.read_text()!=source:
        # Atomic deterministic source cache; separate processes can generate
        # identical modules without exposing a partially written Python file.
        temporary=destination/(name+f'.{os.getpid()}.pending')
        temporary.write_text(source)
        temporary.replace(path)
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    _KERNELS[steps]=module.kernel
    return module.kernel


def is_supported(currents, threshold):
    """Metadata-only eligibility; importing the optional package can fail."""
    if (currents.device.type != 'cuda' or currents.dtype != torch.float32
            or currents.shape[0] not in (3, 4, 7, 15) or threshold.numel() != 1):
        return False
    try:
        import triton
    except ImportError:
        return False
    return True


@torch.no_grad()
def triton_density_vjp(currents, threshold, output_gradient):
    """Launch on eligible, already validated inputs; no hidden fallback.

    Explicit contiguity handles input views and stride-zero cotangents. The
    caller validates shape/device/dtype and preserves the old Torch path.
    """
    import triton
    x = currents.contiguous()
    g = output_gradient.contiguous()
    theta = threshold.to(device=x.device, dtype=x.dtype).contiguous()
    output = torch.empty_like(x)
    neurons = x.numel() // len(x)
    if neurons:
        _kernel(len(x))[(triton.cdiv(neurons, 128),)](
            x, theta, g, output, neurons, 128, num_warps=4,
            enable_fp_fusion=False)
    return output
