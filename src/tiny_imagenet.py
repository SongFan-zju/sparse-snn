"""Tiny-ImageNet source models and deterministic signed-ECL conversion.

Tiny inputs use the CIFAR-style 3x3 ResNet stem. VGG keeps the QCFS author's
13 Conv/BN blocks and two 4096-unit hidden layers, with a fixed global pool.
These are explicitly recorded Tiny adaptations, not ImageNet-1K checkpoints.
"""
from __future__ import annotations
import copy
import hashlib
import json
import random
from pathlib import Path

import torch
from torch import nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.models.resnet34_cifar import ResNet34Cifar
from src.models.vgg16_cifar import VGG16Cifar
from src.conversion.qcfs_ecl_resnet18 import AuthorQCFS, DualThresholdSequence

SEED = 2026100321
MEAN = (.485, .456, .406)
STD = (.229, .224, .225)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(8 * 1024**2), b''): h.update(data)
    return h.hexdigest()


def dataset_manifest(root):
    root = Path(root).resolve()
    classes = sorted(root.joinpath('wnids.txt').read_text().split())
    if len(classes) != 200 or len(set(classes)) != 200:
        raise ValueError('Expected the official Tiny-ImageNet 200 classes')
    class_to_idx = {name: i for i, name in enumerate(classes)}
    train = []
    for name in classes:
        files = sorted((root/'train'/name/'images').glob('*.JPEG'))
        if len(files) != 500: raise ValueError('Expected 500 training images: ' + name)
        train.extend((str(p.relative_to(root)), class_to_idx[name]) for p in files)
    val = []
    per_class = {name: 0 for name in classes}
    for line in (root/'val/val_annotations.txt').read_text().splitlines():
        name, wnid, *_ = line.split()
        if wnid not in class_to_idx: raise ValueError('Unknown validation class')
        path = root/'val/images'/name
        if not path.is_file(): raise FileNotFoundError(path)
        val.append((str(path.relative_to(root)), class_to_idx[wnid]))
        per_class[wnid] += 1
    val.sort()
    if len(val) != 10000 or len({x[0] for x in val}) != 10000 or any(x != 50 for x in per_class.values()):
        raise ValueError('Expected 10000 validation images, 50 per class')
    if len(list((root/'val/images').glob('*.JPEG'))) != 10000:
        raise ValueError('Unexpected validation file count')
    content = json.dumps(dict(classes=classes, train=train, validation=val), separators=(',', ':'))
    return dict(dataset='Tiny-ImageNet-200', image_size=64, root=str(root),
                classes=classes, class_to_idx=class_to_idx, train=train, validation=val,
                manifest_sha256=hashlib.sha256(content.encode()).hexdigest(),
                training_images=100000, validation_images=10000,
                validation_labels='official val_annotations.txt; test split is unlabeled')


class TinyImages(Dataset):
    def __init__(self, manifest, split, training=False):
        self.root = Path(manifest['root'])
        self.items = manifest[split]
        self.training, self.epoch = training, 0
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def __len__(self): return len(self.items)

    def __getitem__(self, index):
        relative, label = self.items[index]
        with Image.open(self.root/relative) as image:
            image = image.convert('RGB')
            if image.size != (64, 64): raise ValueError('Unexpected Tiny image dimensions')
            if self.training:
                rng = random.Random(SEED + 1000003*self.epoch + index)
                image = transforms.functional.pad(image, 8)
                image = transforms.functional.crop(image, rng.randrange(17), rng.randrange(17), 64, 64)
                if rng.random() < .5: image = transforms.functional.hflip(image)
            return self.transform(image), label


class TinyVGG16(VGG16Cifar):
    def __init__(self):
        super().__init__(num_classes=200)
        self.tiny_global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, images):
        for stage in range(1, 6): images = getattr(self, f'layer{stage}')(images)
        return self.classifier(self.tiny_global_pool(images))


def make_model(architecture):
    if architecture == 'resnet34': return ResNet34Cifar(num_classes=200)
    if architecture == 'vgg16': return TinyVGG16()
    raise ValueError(architecture)


def pretrained_initialization(architecture, checkpoint):
    model = make_model(architecture)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if architecture == 'resnet34':
        state = {k.replace('.downsample.', '.shortcut.'): v for k, v in state.items()}
        if tuple(state['conv1.weight'].shape) != (64, 3, 7, 7):
            raise ValueError('Expected official ImageNet ResNet34 initializer')
        state['conv1.weight'] = state['conv1.weight'][:, :, 2:5, 2:5].clone()
        del state['fc.weight'], state['fc.bias']
        missing, extra = model.load_state_dict(state, strict=False)
        if set(missing) != {'fc.weight', 'fc.bias'} or extra:
            raise ValueError((missing, extra))
        note = 'ImageNet stem7x7 center-cropped to Tiny3x3; no stem pool; fresh200-class head'
    else:
        # Map tensor keys directly: no second 138M-parameter model is needed.
        convs = sorted((int(k.split('.')[1]) for k, v in state.items()
                        if k.startswith('features.') and k.endswith('.weight') and v.ndim == 4))
        targets = []
        for stage in range(1, 6):
            layer = getattr(model, f'layer{stage}')
            targets.extend((layer[i], layer[i+1]) for i in range(len(layer)) if isinstance(layer[i], nn.Conv2d))
        if len(convs) != 13 or len(targets) != 13: raise ValueError('Expected 13 VGG blocks')
        for index, (conv, bn) in zip(convs, targets):
            conv.load_state_dict({k: state[f'features.{index}.{k}'] for k in ('weight', 'bias')}, strict=True)
            values = {}
            for key, default in bn.state_dict().items():
                source_key = f'features.{index+1}.{key}'
                if source_key not in state and key == 'num_batches_tracked':
                    values[key] = torch.zeros_like(default)
                else: values[key] = state[source_key]
            bn.load_state_dict(values, strict=True)
        with torch.no_grad():
            model.classifier[1].weight.copy_(state['classifier.0.weight'].reshape(4096, 512, 7, 7).sum((2, 3)))
            model.classifier[1].bias.copy_(state['classifier.0.bias'])
            model.classifier[4].weight.copy_(state['classifier.3.weight'])
            model.classifier[4].bias.copy_(state['classifier.3.bias'])
        note = 'ImageNet VGG16-BN Conv/BN and hidden head transfer; FC6 spatial weights summed for global pool; fresh200-class head; average pools'
    return model, dict(checkpoint=str(Path(checkpoint).resolve()), sha256=sha256(checkpoint),
                       initialization_only=True, dataset='ImageNet-1K pretrained -> Tiny200 adaptation', note=note)


def activation_names(model):
    return [name for name, node in model.named_modules() if isinstance(node, nn.ReLU)]


def install_qcfs(model, levels, thresholds):
    for name in activation_names(model):
        parent, _, child = name.rpartition('.')
        module = AuthorQCFS(T=0, L=levels, thresh=float(thresholds[name]))
        setattr(model.get_submodule(parent), child, module)
    return model


@torch.no_grad()
def calibrate_thresholds(model, batches, levels):
    samples = {name: [] for name in activation_names(model)}
    handles = []
    for name in samples:
        def capture(_node, args, key=name):
            values = args[0].detach().flatten()
            samples[key].append(values[::max(1, values.numel()//4096)][:4096].float().clamp_min(0).cpu())
        handles.append(model.get_submodule(name).register_forward_pre_hook(capture))
    try:
        for images, _labels in batches: model(images.to(next(model.parameters()).device))
    finally:
        for handle in handles: handle.remove()
    thresholds = {}
    for name, pieces in samples.items():
        values = torch.cat(pieces).double()
        step = ((values.mean()+3*values.std())/levels).clamp_min(1e-4)
        for _ in range(32):
            counts = (values/step+.5).floor().clamp(0, levels)
            step = ((values*counts).sum()/counts.square().sum().clamp_min(1)).clamp_min(1e-4)
        thresholds[name] = float(levels*step)
    return thresholds


class TinyECL(nn.Module):
    def __init__(self, source, timesteps, method='ecl'):
        super().__init__()
        if timesteps not in (2, 4): raise ValueError('This task uses T=2 or4')
        self.timesteps = timesteps
        self.source = copy.deepcopy(source)
        converted = 0
        for name, node in list(self.source.named_modules()):
            if not isinstance(node, AuthorQCFS): continue
            if node.T != 0: raise ValueError('Expected a QCFS ANN source')
            if method == 'ecl': replacement = DualThresholdSequence(node.thresh, timesteps, signed=True)
            elif method == 'triangular':
                from src.conversion.ecl_triangular_surrogate import TriangularSequence
                replacement = TriangularSequence(node.thresh)
            elif method == 'density':
                from src.conversion.triangular_density_horizon import TriangularDensityHorizonSequence
                replacement = TriangularDensityHorizonSequence(node.thresh, timesteps)
            else: raise ValueError(method)
            replacement.timesteps = timesteps
            parent, _, child = name.rpartition('.')
            setattr(self.source.get_submodule(parent), child, replacement)
            converted += 1
        if not converted or any(isinstance(n, nn.ReLU) for n in self.source.modules()):
            raise ValueError('Unconverted activation in source')
        if method == 'ecl': self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        super().train(mode)
        for node in self.modules():
            if isinstance(node, nn.modules.batchnorm._BatchNorm): node.eval()
        return self

    def forward(self, images):
        packed = images.unsqueeze(0).expand(self.timesteps, *images.shape).flatten(0, 1)
        return self.source(packed).unflatten(0, (self.timesteps, len(images))).mean(0)
