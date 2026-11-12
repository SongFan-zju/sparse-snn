"""Train a paper recipe with explicit data, weights, and output paths."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli import parser, setup
from src.training import train


def main():
    p = parser(__doc__)
    p.add_argument('--data', required=True, help='Extracted CIFAR directory or its parent')
    p.add_argument('--source', help='ReLU ANN for QAT; QCFS ANN for SNN adaptation')
    p.add_argument('--teacher', help='Frozen ReLU ANN for KD recipes')
    p.add_argument('--thresholds', help='Optional saved channel thresholds for QAT initialization')
    p.add_argument('--output', required=True, help='New experiment directory')
    p.add_argument('--batch-size', type=int, help='Override the paper batch size for available memory')
    p.add_argument('--data-parallel', action='store_true', help='Use all visible GPUs with local BN')
    p.add_argument('--resume', action='store_true', help='Resume output/latest.pt')
    p.add_argument('--checkpoint-every', type=int, default=5)
    args = p.parse_args()
    if args.checkpoint_every < 1:
        p.error('--checkpoint-every must be positive')
    cfg, device = setup(args)
    if args.batch_size is not None:
        cfg['batch_size'] = args.batch_size
    train(cfg, args, device)


if __name__ == '__main__':
    main()
