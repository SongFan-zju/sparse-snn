"""Generate mechanism or corruption-comparison figures."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.plots import phase, corruption


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='kind', required=True)
    mechanism = commands.add_parser('phase')
    mechanism.add_argument('--output', required=True, help='Output directory')
    comparison = commands.add_parser('corruption')
    comparison.add_argument('--phase', required=True, help='JSON from evaluate.py --corruptions')
    comparison.add_argument('--baseline', required=True, help='Matching baseline evaluation JSON')
    comparison.add_argument('--output', required=True, help='Output image or PDF filename')
    args = p.parse_args()
    if args.kind == 'phase':
        phase(args.output)
    else:
        corruption(args.phase, args.baseline, args.output)


if __name__ == '__main__':
    main()
