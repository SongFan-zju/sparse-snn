"""Build trainable or hard-deployment models from portable or original weights."""
import torch

from src.models.resnet18_cifar import resnet18_cifar_ann
from src.models.resnet50_ann import resnet50_cifar_ann
from src.conversion.activation_sites import activation_sites as sites18
from src.conversion.channel_qcfs_source import install_channel_qcfs as install18
from src.conversion.diagonal_phase_adjoint import DiagonalPhaseResNet18
from src.conversion.ecl_triangular_surrogate import TriangularResNet18, TriangularSequence
from src.conversion.integer_cap_phase import IntegerCapResNet18, IntegerCapPhaseResNet18
from src.conversion.joint_phase_adjoint import JointPhaseResNet18
from src.conversion.qcfs_ecl_resnet18 import ECLResNet18
from src.conversion.qcfs_ecl_resnet50 import (
    ECLResNet50, JointPhaseResNet50, activation_sites as sites50, install_channel_qcfs as install50,
)
from src.conversion.recursive_phase_adjoint import RecursivePhaseResNet18


def backbone(cfg):
    build = resnet18_cifar_ann if cfg['architecture'] == 'resnet18' else resnet50_cifar_ann
    return build(num_classes=10 if cfg['dataset'] == 'cifar10' else 100, base_width=cfg['base_width'])


def activation_sites(model, cfg):
    return (sites18 if cfg['architecture'] == 'resnet18' else sites50)(model)


def install_qat(model, thresholds, cfg):
    return (install18 if cfg['architecture'] == 'resnet18' else install50)(model, thresholds)


def convert(source, cfg, *, training=False):
    t, cap = cfg['timesteps'], cfg['max_count']
    if cfg['architecture'] == 'resnet50':
        return JointPhaseResNet50(source) if training else ECLResNet50(source, t, True)
    if cap == 3:
        return (IntegerCapPhaseResNet18 if training else IntegerCapResNet18)(source, t, cap)
    if not training:
        return ECLResNet18(source, t, True)
    if cfg['method'] == 'diagonal':
        return DiagonalPhaseResNet18(source)
    if cfg['method'] == 'triangular':
        model = TriangularResNet18(source)
        model.timesteps = t
        for node in model.modules():
            if isinstance(node, TriangularSequence):
                node.timesteps = t
        return model
    return JointPhaseResNet18(source) if t == 2 else RecursivePhaseResNet18(source, t)


def read_checkpoint(path):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    state = saved.get('model', saved.get('state_dict', saved))
    state = {k.replace('.downsample.', '.shortcut.'): v for k, v in state.items()}
    if not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError('Expected a state_dict or a checkpoint containing model/state_dict')
    return saved, state


def load_ann(path, cfg):
    _, state = read_checkpoint(path)
    model = backbone(cfg)
    model.load_state_dict(state, strict=True)
    return model


def load_qat(path, cfg):
    _, state = read_checkpoint(path)
    model = backbone(cfg)
    thresholds = {name: state[name + '.thresh'] for name, _ in activation_sites(model, cfg)}
    install_qat(model, thresholds, cfg)
    model.load_state_dict(state, strict=True)
    return model


def load_model(path, cfg, *, training=False):
    """SNN checkpoints contain all weights and thresholds: no source ANN needed."""
    if cfg['stage'] == 'ann':
        return load_ann(path, cfg)
    if cfg['stage'] == 'qat':
        return load_qat(path, cfg)
    saved, state = read_checkpoint(path)
    recorded = saved.get('config', {})
    for key in ('timesteps', 'max_count', 'max_absolute_count_per_step'):
        expected = cfg['timesteps'] if key == 'timesteps' else cfg['max_count']
        if key in recorded and recorded[key] != expected:
            raise ValueError(f'Checkpoint {key}={recorded[key]} differs from recipe {expected}')
    source = backbone(cfg)
    thresholds = {name: state['source.' + name + '.threshold']
                  for name, _ in activation_sites(source, cfg)}
    model = convert(install_qat(source, thresholds, cfg), cfg, training=training)
    model.load_state_dict(state, strict=True)
    return model
