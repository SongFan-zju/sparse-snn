"""Portable runtime configuration, random streams and checkpoint I/O."""
import copy
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch


def configure(device, seed, threads=4):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(threads)
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(seed)
    return device


def generators(seed):
    return dict(order=torch.Generator().manual_seed(seed),
                augment=torch.Generator().manual_seed(seed + 1009),
                mix=np.random.default_rng(seed + 2003))


def rng_state(streams):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
                order=streams['order'].get_state(), augment=streams['augment'].get_state(),
                mix=copy.deepcopy(streams['mix'].bit_generator.state))


def restore_rng(saved, streams):
    random.setstate(saved['python'])
    np.random.set_state(saved['numpy'])
    torch.set_rng_state(saved['torch'])
    if saved['cuda'] and torch.cuda.is_initialized():
        torch.cuda.set_rng_state_all(saved['cuda'])
    streams['order'].set_state(saved['order'])
    streams['augment'].set_state(saved['augment'])
    streams['mix'].bit_generator.state = saved['mix']


def cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value, *, json_format=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.pending')
    with temporary.open('w' if json_format else 'wb') as handle:
        if json_format:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')
        else:
            torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def report(value, path=None):
    if path:
        save(path, value, json_format=True)
    print(json.dumps(value, sort_keys=True, allow_nan=False), flush=True)
