"""Sampled per-channel three-level least-squares QCFS initialization."""
import torch

from src.data import normalize
from src.models.factory import activation_sites


@torch.no_grad()
def calibrate(model, raw, indices, cfg, device):
    model = model.to(device).eval()
    generator = torch.Generator().manual_seed(cfg['seed'])
    samples, handles = {}, []
    for name, _ in activation_sites(model, cfg):
        samples[name] = []

        def capture(module, inputs, name=name):
            values = inputs[0].detach().relu().flatten(2)
            positions = torch.randint(values.shape[2], (len(values), 16), generator=generator).to(device)
            chosen = values.gather(2, positions[:, None].expand(-1, values.shape[1], -1))
            samples[name].append(chosen.permute(1, 0, 2).flatten(1).cpu())

        handles.append(model.get_submodule(name).register_forward_pre_hook(capture))
    try:
        size = cfg.get('calibration_batch_size', 64)
        for offset in range(0, len(indices), size):
            model(normalize(raw[indices[offset:offset + size]], cfg['dataset']).to(device))
    finally:
        for handle in handles:
            handle.remove()
    thresholds = {}
    for name, parts in samples.items():
        values = torch.cat(parts, 1).to(device)
        step = ((values.mean(1) + 3 * values.std(1)) / 2).clamp_min(1e-6)
        for _ in range(32):
            count = (values / step[:, None] + .5).floor().clamp(0, 2)
            step = ((values * count).sum(1) / count.square().sum(1).clamp_min(1)).clamp_min(1e-6)
        thresholds[name] = (2 * step).cpu()
    return thresholds
