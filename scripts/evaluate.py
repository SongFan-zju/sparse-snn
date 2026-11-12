"""Evaluate one checkpoint; no teacher or original training directory is needed."""
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.cli import parser, setup
from src.data import load_cifar, partitions
from src.evaluation import evaluate, evaluate_corruptions
from src.models.factory import convert, load_model, load_qat
from src.runtime import digest, report


def main():
    p = parser(__doc__)
    p.add_argument('--checkpoint', required=True)
    data = p.add_mutually_exclusive_group(required=True)
    data.add_argument('--data', help='Extracted CIFAR directory or its parent')
    data.add_argument('--corruptions', help='Extracted CIFAR-100-C directory')
    p.add_argument('--split', choices=('test', 'selection', 'confirmation', 'calibration'), default='test')
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--unadapted', action='store_true', help='Convert a QAT checkpoint without SNN adaptation')
    p.add_argument('--output', help='Optional JSON report')
    args = p.parse_args()
    cfg, device = setup(args)
    if args.unadapted and cfg['stage'] != 'snn':
        p.error('--unadapted requires an SNN recipe and a QAT checkpoint')
    model = (convert(load_qat(args.checkpoint, cfg), cfg) if args.unadapted else
             load_model(args.checkpoint, cfg)).to(device).eval().requires_grad_(False)
    started = time.monotonic()
    if args.corruptions:
        if cfg['dataset'] != 'cifar100' or args.split != 'test':
            p.error('CIFAR100-C evaluation requires a cifar100 recipe and --split test')
        metrics = evaluate_corruptions(model, args.corruptions, device, args.batch_size)
    else:
        raw, labels = load_cifar(args.data, cfg['dataset'], 'test' if args.split == 'test' else 'train')
        if args.split == 'test':
            indices = np.arange(len(raw))
        else:
            roles = partitions(cfg['dataset'], len(raw), args.split_file, cfg['seed'])
            if args.split not in roles:
                p.error('CIFAR10 has no internal held-out selection or confirmation partition')
            indices = roles[args.split]
        metrics = evaluate(model, raw, labels, indices, cfg['dataset'], device, args.batch_size)
    report(dict(experiment=args.experiment, config=cfg, checkpoint_sha256=digest(args.checkpoint),
                split='cifar100c' if args.corruptions else args.split, unadapted=args.unadapted,
                device=str(device), batch_size=args.batch_size, metrics=metrics,
                seconds=time.monotonic() - started), args.output)


if __name__ == '__main__':
    main()
