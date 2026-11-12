"""Hard inference on clean CIFAR and all 75 CIFAR100-C cases."""
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from src.data import CORRUPTIONS, normalize


@torch.inference_mode()
def evaluate(model, raw, labels, indices, dataset, device, batch_size):
    model.eval()
    correct, loss = 0, 0.
    for offset in range(0, len(indices), batch_size):
        selected = indices[offset:offset + batch_size]
        logits = model(normalize(raw[selected], dataset).to(device))
        targets = torch.from_numpy(np.array(labels[selected], copy=True)).to(device)
        if not torch.isfinite(logits).all():
            raise RuntimeError('Nonfinite evaluation logits')
        correct += int((logits.argmax(1) == targets).sum())
        loss += float(F.cross_entropy(logits, targets, reduction='sum'))
    if len(indices) == 0:
        raise ValueError('Evaluation partition is empty')
    return dict(top1=100 * correct / len(indices), correct=correct,
                count=len(indices), ce=loss / len(indices))


def evaluate_corruptions(model, root, device, batch_size):
    root = Path(root)
    labels = np.load(root / 'labels.npy', mmap_mode='r')
    if labels.shape not in ((10000,), (50000,)):
        raise ValueError('CIFAR100-C labels must contain 10000 or 50000 entries')
    rows = {}
    for name in CORRUPTIONS:
        images = np.load(root / (name + '.npy'), mmap_mode='r')
        if images.shape != (50000, 32, 32, 3):
            raise ValueError(f'{name}: expected five severities of 10000 images')
        rows[name] = []
        for severity in range(5):
            start, stop = severity * 10000, (severity + 1) * 10000
            raw = images[start:stop].transpose(0, 3, 1, 2)
            y = labels[start:stop] if len(labels) == 50000 else labels
            metrics = evaluate(model, raw, y, np.arange(10000), 'cifar100', device, batch_size)
            rows[name].append(dict(severity=severity + 1, **metrics))
    correct = sum(row['correct'] for cases in rows.values() for row in cases)
    return dict(top1=100 * correct / 750000, correct=correct, count=750000,
                cases=75, corruptions=rows)
