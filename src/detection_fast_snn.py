"""Use the author's Fast-SNN YOLOv2-R34 with fixed binary ECL gradients.

The upstream checkout stays external: pass its `object detection` directory.
Only runtime compatibility (old TorchVision/NumPy names) is patched on import.
The ANN topology, label assignment, detection losses and NMS remain upstream.
"""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import torch
from torch import nn

from src.conversion.general_phase_adjoint import TriangularHorizonSequence
from src.conversion.qcfs_ecl_resnet18 import DualThresholdSequence
from src.conversion.triangular_density_horizon import TriangularDensityHorizonSequence
from src.conversion.triangular_density_phase import TriangularDensityPhaseSequence


_PREFIX = '_sparse_fast_snn_detection'
_RUNTIME = None


def upstream_runtime(path):
    """Load only required author modules under an isolated package name."""
    global _RUNTIME
    path = Path(path).resolve()
    if _RUNTIME is not None:
        if _RUNTIME.path != path:
            raise ValueError('A process may use only one upstream checkout')
        return _RUNTIME
    if not (path / 'models/yolov2_r34.py').is_file():
        raise ValueError('Expected the Fast-SNN object detection directory')
    for suffix in ('', '.backbone', '.models', '.utils', '.data'):
        package = ModuleType(_PREFIX + suffix)
        package.__path__ = [str(path / suffix.lstrip('.').replace('.', '/'))]
        sys.modules[package.__name__] = package

    def load(name, filename):
        fullname = _PREFIX + '.' + name
        filename = path / filename
        spec = importlib.util.spec_from_file_location(fullname, filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[fullname] = module
        source = filename.read_text()
        for package in ('data', 'backbone'):
            source = source.replace(f'from {package} ', f'from {_PREFIX}.{package} ')
            source = source.replace(f'from {package}.', f'from {_PREFIX}.{package}.')
        source = source.replace('import tools', f'from {_PREFIX} import tools')
        source = source.replace('from torchvision.models.resnet import model_urls',
                                "model_urls = {'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'}")
        source = source.replace('dtype=np.int)', 'dtype=np.int64)')
        source = source.replace('.astype(np.bool)', '.astype(bool)')
        source = source.replace('if dets == []:', 'if len(dets) == 0:')
        source = source.replace('rec = -1.\n            prec = -1.\n            ap = -1.',
                                'rec = np.empty(0)\n            prec = np.empty(0)\n            ap = 0.')
        exec(compile(source, str(filename), 'exec'), module.__dict__)
        parent, _, child = fullname.rpartition('.')
        setattr(sys.modules[parent], child, module)
        return module

    quant = load('backbone.quant_layer', 'backbone/quant_layer.py')
    backbone = load('backbone.myresnet', 'backbone/myresnet.py')
    load('data.voc0712', 'data/voc0712.py')
    load('data.coco2017', 'data/coco2017.py')
    data = load('data', 'data/__init__.py')
    tools = load('tools', 'tools.py')
    model = load('models.yolov2_r34', 'models/yolov2_r34.py')
    augment = load('utils.augmentations', 'utils/augmentations.py')
    evaluator = load('utils.vocapi_evaluator', 'utils/vocapi_evaluator.py')
    config = load('data.config', 'data/config.py')
    _RUNTIME = SimpleNamespace(path=path, quant=quant, backbone=backbone,
                               data=data, tools=tools, model=model, augment=augment,
                               evaluator=evaluator, config=config)
    return _RUNTIME


class PhaseDetectionBackbone(nn.Module):
    """Vectorized [T,B,...] signed-ECL trajectory, then one final prediction."""
    def __init__(self, source, quant_type, timesteps, method):
        super().__init__()
        if not isinstance(timesteps, int) or timesteps < 1:
            raise ValueError('T must be a positive integer')
        if method not in ('hard', 'triangular', 'phase_triangular_density'):
            raise ValueError('Expected hard, triangular or phase_triangular_density')
        self.source = copy.deepcopy(source)
        self.timesteps = timesteps
        self.method = method
        sites = [(name, node) for name, node in self.source.named_modules()
                 if isinstance(node, quant_type)]
        if len(sites) != 38:
            raise ValueError(f'Expected 38 author YOLOv2-R34 activations, got {len(sites)}')
        for name, node in sites:
            threshold = node.act_alpha.detach().clone()
            if not bool(torch.isfinite(threshold).all() and (threshold > 0).all()):
                raise ValueError('Source activation thresholds must be positive and finite')
            if method == 'hard':
                sequence = DualThresholdSequence(threshold, timesteps, True)
            elif method == 'triangular':
                sequence = TriangularHorizonSequence(threshold, timesteps)
            elif timesteps == 2:
                sequence = TriangularDensityPhaseSequence(threshold)
            else:
                sequence = TriangularDensityHorizonSequence(threshold, timesteps)
            parent, _, child = name.rpartition('.')
            setattr(self.source.get_submodule(parent), child, sequence)
        self.requires_grad_(method != 'hard')

    def forward(self, images):
        # The author's SNN uses [B,T,...]; the source ANN is a normal 4D graph.
        # Folding time first consistently matches all shared-phase operators.
        x = images.unsqueeze(0).expand(self.timesteps, *images.shape).flatten(0, 1)
        x = self.source.relu0(self.source.layer0(x))
        x = self.source.layer1(x)
        x = self.source.layer2(x)
        c4 = self.source.layer3(x)
        c5 = self.source.layer4(c4)
        c5 = self.source.head3(self.source.head2(self.source.head1(c5)))
        c4 = self.source.reorg(self.source.route_layer(c4))
        features = self.source.head4(torch.cat((c4, c5), dim=1))
        features = features.unflatten(0, (self.timesteps, len(images))).mean(0)
        return self.source.pred(features)


def build_detector(runtime, checkpoint, *, bit=2, timesteps=3, method='ann',
                   size=416, device='cpu', trainable=False):
    """Load the actual public QAT checkpoint before replacing activations."""
    cfg = runtime.config.yolov2_r50_cfg  # Author also uses this anchor config for R34.
    detector = runtime.model.YOLOv2R34(
        device=torch.device('cpu'), input_size=size, num_classes=20,
        trainable=trainable, anchor_size=cfg['anchor_size_voc'], bit=bit, spike=method == 'author')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = saved.get('model', saved.get('state_dict', saved))
    detector.load_state_dict(state, strict=True)
    if method not in ('ann', 'author'):
        detector.backbone = PhaseDetectionBackbone(
            detector.backbone, runtime.quant.QuantReLU, timesteps, method)
    detector.to(device)
    detector.device = torch.device(device)
    detector.set_grid(size)
    return detector


def make_evaluator(runtime, data_root, output, device, size=416):
    """Retain author's VOC07 AP50 evaluation, isolate each run's output files."""
    evaluator = runtime.evaluator.VOCAPIEvaluator(
        data_root=str(data_root), img_size=size, device=torch.device(device),
        transform=runtime.data.BaseTransform(size))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    evaluator.devkit_path = str(output)
    evaluator.output_dir = str(output / 'precision_recall')
    Path(evaluator.output_dir).mkdir(exist_ok=True)
    return evaluator
