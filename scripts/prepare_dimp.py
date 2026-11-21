#!/usr/bin/env python3
"""Prepare DiMP18 quantization/conversion interfaces without training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from src.tracking_dimp import (DimpFeatureBackbone, calibrate_dimp_thresholds,
                               install_dimp_qcfs, load_official_dimp18,
                               split_dimp18_activations)


def load_pair_images(pair, size):
    """Use the original crop geometry and ImageNet RGB normalization."""
    import cv2
    from ltr.data.processing_utils import sample_target
    images = []
    for branch in ('template', 'search'):
        row = pair[branch]
        image = cv2.imread(row['image'])
        if image is None:
            raise ValueError('Cannot read ' + row['image'])
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        crop, _ = sample_target(rgb, torch.tensor(row['bbox_xywh']), 5.0, size)
        value = torch.from_numpy(crop.copy()).permute(2, 0, 1).float() / 255
        mean = value.new_tensor([.485, .456, .406])[:, None, None]
        std = value.new_tensor([.229, .224, .225])[:, None, None]
        images.append((value - mean) / std)
    return torch.stack(images)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--L', type=int, default=4)
    parser.add_argument('--T', type=int, default=4)
    parser.add_argument('--calibration-pairs', type=int, default=2)
    parser.add_argument('--image-size', type=int, default=128)
    args = parser.parse_args()
    if args.calibration_pairs < 1 or args.image_size < 32:
        parser.error('Positive calibration count and image size >=32 are required')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a fresh output directory to preserve preparation results')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(2026100321)
    if torch.cuda.is_initialized():
        raise RuntimeError('This preparation script is CPU-only')
    model, provenance = load_official_dimp18(args.upstream, args.checkpoint)
    split = split_dimp18_activations(model.feature_extractor)
    pairs = []
    with args.pairs.open() as stream:
        for line in stream:
            if len(pairs) >= args.calibration_pairs:
                break
            pairs.append(json.loads(line))
    if len(pairs) != args.calibration_pairs:
        raise ValueError('Not enough fixed training pairs')
    batches = [load_pair_images(pair, args.image_size) for pair in pairs]
    with torch.no_grad():
        native = model.extract_backbone_features(batches[0])
        independent = split(batches[0], ['layer2', 'layer3'])
        ann_parity = {name: bool(torch.equal(value, independent[name]))
                      for name, value in native.items()}
    if not all(ann_parity.values()):
        raise RuntimeError('Splitting ReLU calls changed ANN features')
    calibration = calibrate_dimp_thresholds(split, batches)
    source = install_dimp_qcfs(split, args.L, calibration['thresholds'])
    report = dict(status='conversion_interfaces_checked_source_training_pending',
                  checkpoint=provenance, L=args.L, T=args.T, seed=2026100321,
                  calibration=calibration, ann_split_bitwise_equal=ann_parity,
                  pairs=str(args.pairs.resolve()), image_size=args.image_size,
                  pair_manifest_sha256=hashlib.sha256(args.pairs.read_bytes()).hexdigest(),
                  method_hyperparameters=['L', 'T'], methods={},
                  full_native_head_check='See ../native_cuda_smoke.json; separate check',
                  no_training=True, source_is_trained=False)
    reference = DimpFeatureBackbone(source, args.T, 'hard')
    with torch.no_grad():
        hard = reference(batches[0])
    for method in ('triangular', 'density'):
        converted = DimpFeatureBackbone(source, args.T, method).train()
        buffers = {name: value.clone() for name, value in converted.named_buffers()}
        features = converted(batches[0])
        equal = all(torch.equal(hard[name], value) for name, value in features.items())
        if not equal:
            raise RuntimeError('Surrogate method changed hard features')
        loss = sum((value.square() + value).mean() for value in features.values())
        loss.backward()
        gradients = {name: parameter.grad for name, parameter in converted.named_parameters()
                     if parameter.requires_grad}
        finite = all(gradient is not None and bool(torch.isfinite(gradient).all())
                     for gradient in gradients.values())
        stable = all(torch.equal(buffers[name], value) for name, value in converted.named_buffers())
        repeated = converted(batches[0])
        independent_images = all(torch.allclose(features[name][:1], converted(batches[0][:1])[name],
                                               rtol=1e-5, atol=1e-6) for name in features)
        reset = all(torch.equal(features[name], repeated[name]) for name in features)
        first_grad = gradients['source.conv1.weight']
        report['methods'][method] = dict(hard_feature_parity=equal,
            all_active_gradients_present_and_finite=finite,
            first_conv_gradient_l1=float(first_grad.abs().sum()) if first_grad is not None else None,
            source_buffers_unchanged=stable, repeat_call_equal=reset,
            independent_image_parity=independent_images,
            feature_shapes={name: list(value.shape) for name, value in features.items()})
        if not all((finite, stable, reset, independent_images)):
            raise RuntimeError('Converted feature check failed: ' + method)
    torch.save(dict(backbone=source.state_dict(), L=args.L,
                    provenance=provenance, calibration=calibration,
                    source_is_trained=False, seed=2026100321),
               args.output/'qcfs_initialization.pt')
    report['cuda_initialized'] = torch.cuda.is_initialized()
    (args.output/'readiness.json').write_text(json.dumps(report, indent=2)+'\n')
    (args.output/'selected_pairs.json').write_text(json.dumps(pairs, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
