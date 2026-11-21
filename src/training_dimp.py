"""Shared full-model checkpoint interface for DiMP18 training.

The pilot QCFS source has a task-adaptation pass, not a qualified source model.
Native classifier/initializer/IoU heads are preserved for every method.
"""
from __future__ import annotations
import copy
import hashlib
from pathlib import Path
import torch
from torch import nn
from src.tracking_dimp import (DimpFeatureBackbone, install_dimp_qcfs,
                              load_official_dimp18, split_dimp18_activations)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREP = ROOT / 'artifacts/dimp18_preparation_20261119'
DEFAULT_OFFICIAL = ROOT / 'artifacts/tracking_framework_candidates_20261119/dimp18.pth'
DEFAULT_UPSTREAM = ROOT / 'artifacts/tracking_framework_candidates_20261119/upstream/pytracking-7eb9e74bd3d40e29dbcec444902237da13de247b'


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def freeze_batchnorm(net):
    """Freeze statistics globally, while leaving affine parameters trainable."""
    for module in net.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
    return net


def initialized_model(initialization=DEFAULT_PREP/'conversion/qcfs_initialization.pt',
                      upstream=DEFAULT_UPSTREAM, official_checkpoint=DEFAULT_OFFICIAL):
    initialization = Path(initialization).resolve()
    raw = torch.load(initialization, map_location='cpu', weights_only=False)
    net, provenance = load_official_dimp18(upstream, official_checkpoint)
    source = install_dimp_qcfs(split_dimp18_activations(net.feature_extractor),
                               raw['L'], raw['calibration']['thresholds'])
    source.load_state_dict(raw['backbone'], strict=True)
    net.feature_extractor = DimpFeatureBackbone(source, T=4, method='source')
    metadata = dict(format='dimp18_pilot_v1', stage='calibration_initialization',
                    method='source', L=raw['L'], T=4,
                    official_upstream=str(Path(upstream).resolve()),
                    official_checkpoint=str(Path(official_checkpoint).resolve()),
                    official_checkpoint_sha256=provenance['sha256'],
                    source_initialization=str(initialization),
                    source_initialization_sha256=file_sha256(initialization),
                    source_checkpoint_sha256=file_sha256(initialization),
                    calibration=copy.deepcopy(raw['calibration']),
                    source_is_trained=False, source_quality_qualified=False,
                    seed=2026100321)
    return freeze_batchnorm(net.eval()), metadata


def model_from_checkpoint(path, device='cpu'):
    """Return (full native model, metadata), strict-loaded and eval/BN frozen.

    This reads our complete pilot checkpoint, not a bare backbone. Only the
    local official DiMP checkpoint and stored calibration metadata are needed.
    """
    raw = torch.load(Path(path), map_location='cpu', weights_only=False)
    metadata = copy.deepcopy(raw['metadata'])
    if metadata.get('format') not in ('dimp18_pilot_v1', 'dimp18_training_v1'):
        raise ValueError('Expected a complete DiMP18 training checkpoint')
    net, _ = load_official_dimp18(metadata['official_upstream'], metadata['official_checkpoint'])
    source = install_dimp_qcfs(split_dimp18_activations(net.feature_extractor),
                               int(metadata['L']), metadata['calibration']['thresholds'])
    net.feature_extractor = DimpFeatureBackbone(source, int(metadata['T']), metadata['method'])
    net.load_state_dict(raw['model'], strict=True)
    net = freeze_batchnorm(net.to(device).eval())
    metadata.update(checkpoint=str(Path(path).resolve()), checkpoint_sha256=file_sha256(path),
                    completed_epochs=raw['progress']['completed_epochs'],
                    optimizer_steps=raw['progress']['optimizer_steps'],
                    actual_images=raw['progress']['actual_images'])
    return net, metadata


def warm_started_model(path, device='cpu'):
    """Inherit complete weights only; caller creates fresh progress/Adam/RNG."""
    net, metadata = model_from_checkpoint(path, device)
    prior_images = int(metadata.get('actual_images', 0))
    prior_extra = int(metadata.get('warmup_extra_training_images', 0))
    prior_stage = metadata.get('stage')
    for key in ('checkpoint', 'checkpoint_sha256', 'completed_epochs',
                'optimizer_steps', 'actual_images'):
        metadata.pop(key, None)
    metadata.update(warm_start_checkpoint=str(Path(path).resolve()),
                    warm_start_checkpoint_sha256=file_sha256(path),
                    warm_start_prior_stage=prior_stage,
                    warm_start_prior_training_images=prior_images,
                    warmup_extra_training_images=prior_extra + prior_images,
                    warm_start_optimizer_inherited=False,
                    warm_start_cursor_inherited=False,
                    source_quality_qualified=False)
    return net, metadata


def converted_model_from_source(path, method, T=4):
    if method not in ('triangular', 'density'):
        raise ValueError('Expected a matched triangular or density arm')
    net, metadata = model_from_checkpoint(path)
    if metadata['method'] != 'source':
        raise ValueError('SNN initialization must use the common adapted QCFS source')
    if not metadata.get('source_is_trained'):
        raise ValueError('Task-adapt the initialized source before starting SNN arms')
    net.feature_extractor = DimpFeatureBackbone(net.feature_extractor.source, T, method)
    metadata['common_source_training_images'] = metadata.get('actual_images', 0)
    for key in ('checkpoint', 'checkpoint_sha256', 'completed_epochs',
                'optimizer_steps', 'actual_images'):
        metadata.pop(key, None)
    metadata.update(stage='pilot_snn_finetuning', method=method, T=T,
                    source_checkpoint=str(Path(path).resolve()),
                    source_checkpoint_sha256=file_sha256(path),
                    source_quality_qualified=False)
    return freeze_batchnorm(net.eval()), metadata
