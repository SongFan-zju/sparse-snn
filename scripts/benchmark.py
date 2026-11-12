"""Measure one hard SNN's dense PyTorch runtime and signed-event activity."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from src.cli import parser, setup
from src.cost import collect_activity, measure
from src.data import load_cifar, normalize, partitions
from src.models.factory import load_model
from src.runtime import digest, report


def main():
    p = parser(__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--samples', type=int, default=1000)
    p.add_argument('--output', help='Optional JSON report')
    args = p.parse_args()
    cfg, device = setup(args)
    if cfg['stage'] != 'snn':
        p.error('Event cost measurement requires an SNN recipe')
    raw, _ = load_cifar(args.data, cfg['dataset'], 'train')
    # Match the reported sets: C100 calibration partition; C10 training indices 0:1000.
    indices = (partitions(cfg['dataset'], len(raw), args.split_file, cfg['seed'])['calibration']
               if cfg['dataset'] == 'cifar100' else np.arange(min(1000, len(raw))))
    if not 1 <= args.samples <= len(indices):
        p.error('--samples must fit within the 1000-image measurement set')
    indices = indices[:args.samples]
    images = normalize(raw[indices], cfg['dataset']).to(device)
    del raw
    batches = list(images.split(args.batch_size))
    model = load_model(args.checkpoint, cfg).to(device).eval().requires_grad_(False)
    timing = measure(model, batches, device)
    spec = dict(max_count=cfg['max_count'], event_nodes=17 if cfg['architecture'] == 'resnet18' else 49)
    activity = collect_activity(model, batches, spec)
    report(dict(experiment=args.experiment, config=cfg, checkpoint_sha256=digest(args.checkpoint),
                device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
                torch_version=str(torch.__version__), indices=indices.tolist(), timing=timing, activity=activity,
                scope='Dense FP32 PyTorch simulation. Amortized latency with preloaded inputs; activity hooks are outside timing. Not sparse-kernel speed or measured energy.'), args.output)


if __name__ == '__main__':
    main()
