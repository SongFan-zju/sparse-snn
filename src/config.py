"""Paper recipes, independent of local experiment directories."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'configs/experiments.json'
DEFAULT_SPLIT = ROOT / 'configs/cifar100_split.json'


def recipe(name, path=DEFAULT_CONFIG):
    spec = json.loads(Path(path).read_text())
    if name not in spec['experiments']:
        raise ValueError(f'Unknown experiment {name}; choose from {list(spec["experiments"])}')
    cfg = dict(spec['defaults'], **spec['experiments'][name], experiment=name)
    if cfg['dataset'] not in ('cifar10', 'cifar100'):
        raise ValueError('Expected cifar10 or cifar100')
    if cfg['architecture'] not in ('resnet18', 'resnet50'):
        raise ValueError('Expected resnet18 or resnet50')
    if cfg['stage'] not in ('ann', 'qat', 'snn'):
        raise ValueError('Expected ann, qat or snn stage')
    if cfg['method'] not in ('phase', 'triangular', 'diagonal'):
        raise ValueError('Unknown surrogate method')
    if cfg['timesteps'] not in (2, 4) or cfg['max_count'] not in (1, 3):
        raise ValueError('Paper recipes use T=2/4 and M=1/3')
    if cfg['stage'] == 'snn':
        if cfg['architecture'] == 'resnet50' and (cfg['timesteps'], cfg['max_count'], cfg['method']) != (2, 1, 'phase'):
            raise ValueError('The ResNet50 paper recipe uses M1/T2 phase gradients')
        if cfg['method'] == 'diagonal' and (cfg['timesteps'], cfg['max_count']) != (2, 1):
            raise ValueError('Diagonal control requires M1/T2')
        if cfg['max_count'] == 3 and cfg['method'] != 'phase':
            raise ValueError('The M3 recipe uses phase gradients')
    return cfg
