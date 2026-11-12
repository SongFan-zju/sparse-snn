"""CIFAR loading, fixed partitions, and the paper's pixel-space augmentation."""
import json
import math
from pathlib import Path
import pickle

import numpy as np
import torch
from torch.nn import functional as F

NORMALIZATION = {
    'cifar100': ((.5071, .4867, .4408), (.2675, .2565, .2761)),
    'cifar10': ((.4914, .4822, .4465), (.2023, .1994, .201)),
}
CORRUPTIONS = ('gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
               'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
               'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression')


def load_cifar(root, dataset, split='train'):
    if split not in ('train', 'test'):
        raise ValueError(split)
    folder = Path(root)
    subdir = 'cifar-100-python' if dataset == 'cifar100' else 'cifar-10-batches-py'
    if (folder / subdir).is_dir():
        folder = folder / subdir
    names = [split] if dataset == 'cifar100' else (
        [f'data_batch_{i}' for i in range(1, 6)] if split == 'train' else ['test_batch'])
    images, labels = [], []
    for name in names:
        with (folder / name).open('rb') as handle:
            part = pickle.load(handle, encoding='latin1')
        images.append(part['data'].reshape(-1, 3, 32, 32))
        labels.extend(part['fine_labels' if dataset == 'cifar100' else 'labels'])
    raw, targets = np.concatenate(images), np.asarray(labels, dtype=np.int64)
    if raw.dtype != np.uint8 or len(raw) != len(targets) or len(raw) == 0:
        raise ValueError('Invalid CIFAR image/label arrays')
    return raw, targets


def partitions(dataset, count, split_path, seed):
    if dataset == 'cifar10':
        return dict(training=np.arange(count), calibration=torch.randperm(
            count, generator=torch.Generator().manual_seed(seed))[:1000].numpy())
    spec = json.loads(Path(split_path).read_text())
    # Accept the public compact partition and the original experiment manifest.
    spec = spec.get('roles', spec)
    roles = {name: np.asarray(value.get('indices') if isinstance(value, dict) else value, dtype=np.int64)
             for name, value in spec.items() if name in ('training', 'calibration', 'selection', 'confirmation')}
    if set(roles) != {'training', 'calibration', 'selection', 'confirmation'}:
        raise ValueError('CIFAR100 requires four explicit partition roles')
    joined = np.concatenate(list(roles.values()))
    if not np.array_equal(np.sort(joined), np.arange(count)):
        raise ValueError('Partition must cover the training set exactly once')
    return roles


def normalize(images, dataset):
    mean, std = NORMALIZATION[dataset]
    x = torch.as_tensor(np.array(images, copy=True)).float() / 255.
    return (x - x.new_tensor(mean).view(1, 3, 1, 1)) / x.new_tensor(std).view(1, 3, 1, 1)


def augment(raw, indices, generator, device, dataset):
    mean, std = NORMALIZATION[dataset]
    x = torch.from_numpy(np.array(raw[indices], copy=True)).to(device).float() / 255.
    x = F.pad(x, (4, 4, 4, 4))
    offsets = torch.randint(9, (len(x), 2), generator=generator).to(device)
    flip = (torch.rand(len(x), generator=generator) < .5).to(device)
    rows = offsets[:, 0, None, None] + torch.arange(32, device=device)[None, :, None]
    cols = offsets[:, 1, None, None] + torch.arange(32, device=device)[None, None, :]
    x = x.permute(0, 2, 3, 1)[torch.arange(len(x), device=device)[:, None, None], rows, cols]
    x = x.permute(0, 3, 1, 2).contiguous()
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    return (x - x.new_tensor(mean).view(1, 3, 1, 1)) / x.new_tensor(std).view(1, 3, 1, 1)


def mixed_batch(raw, labels, indices, streams, device, dataset, mode=None):
    x = augment(raw, indices, streams['augment'], device, dataset)
    y = torch.from_numpy(labels[indices].copy()).to(device)
    rng = streams['mix']
    mode = mode or ('cutmix' if rng.random() < .5 else 'mixup')
    permutation = torch.as_tensor(rng.permutation(len(x)), device=device)
    if mode == 'mixup':
        lam = float(rng.beta(.2, .2))
        x = lam * x + (1 - lam) * x[permutation]
    elif mode == 'cutmix':
        proposed = float(rng.beta(1., 1.))
        rh = rw = int(32 * math.sqrt(1 - proposed))
        cy, cx = int(rng.integers(32)), int(rng.integers(32))
        y1, y2 = max(0, cy - rh // 2), min(32, cy + rh // 2)
        x1, x2 = max(0, cx - rw // 2), min(32, cx + rw // 2)
        mixed = x.clone()
        mixed[:, :, y1:y2, x1:x2] = x[permutation, :, y1:y2, x1:x2]
        x = mixed
        lam = 1. - (y2 - y1) * (x2 - x1) / (32 * 32)
    else:
        raise ValueError(mode)
    return x, y, y[permutation], lam, mode
