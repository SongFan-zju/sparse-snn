"""Shared command-line options; model/data paths are always explicit."""
import argparse
import json

from src.config import DEFAULT_CONFIG, DEFAULT_SPLIT, recipe
from src.runtime import configure


def parser(description):
    names = ', '.join(json.loads(DEFAULT_CONFIG.read_text())['experiments'])
    result = argparse.ArgumentParser(description=description, epilog='Paper presets: ' + names)
    result.add_argument('experiment', help='Preset name in the configuration file')
    result.add_argument('--config', default=str(DEFAULT_CONFIG), help='Recipe JSON')
    result.add_argument('--device', default='cpu', help='cpu or a logical CUDA device, e.g. cuda:0')
    result.add_argument('--threads', type=int, default=4)
    result.add_argument('--split-file', default=str(DEFAULT_SPLIT), help='CIFAR100 training partitions')
    return result


def setup(args):
    if args.threads < 1 or getattr(args, 'batch_size', 1) is not None and getattr(args, 'batch_size', 1) < 1:
        raise ValueError('Thread and batch counts must be positive')
    cfg = recipe(args.experiment, args.config)
    return cfg, configure(args.device, cfg['seed'], args.threads)
