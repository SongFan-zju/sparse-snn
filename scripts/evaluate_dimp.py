#!/usr/bin/env python3
"""Evaluate fixed complete SOT tracks through the original DiMP tracker."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--vendor', type=Path)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--model-kind', choices=('ann', 'trained'), default='trained')
    parser.add_argument('--direct-snn', action='store_true',
                        help='Evaluate hard ECL directly from the common QCFS source')
    parser.add_argument('--tracks', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image-size', type=int, default=288)
    parser.add_argument('--memory-cap-gib', type=float, default=2)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('0', '2', '3'):
        raise ValueError('Expose exactly one authorized physical GPU')
    if args.image_size < 32 or args.image_size % 16 or args.memory_cap_gib <= 0:
        parser.error('Positive cap and image size divisible by16 are required')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('A fresh output directory is required')
    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output/'process.json', dict(pid=os.getpid(),
        start_ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19],
        physical_GPU=os.environ['CUDA_VISIBLE_DEVICES']))
    if args.vendor:
        sys.path.insert(0, str(args.vendor.resolve()))
    sys.path.insert(0, str(args.upstream.resolve()))
    import cv2
    import numpy as np
    import torch
    from pytracking.tracker.dimp.dimp import DiMP
    from pytracking.parameter.dimp.dimp18 import parameters
    from pytracking.features.net_wrappers import NetWithBackbone
    from src.dimp_evaluation import overlap_and_distance, sot_metrics
    from src.tracking_dimp import load_official_dimp18
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    # The original ROI backward uses atomics for online box refinement.
    torch.use_deterministic_algorithms(False)
    torch.cuda.set_per_process_memory_fraction(args.memory_cap_gib*1024**3 /
        torch.cuda.get_device_properties(0).total_memory, 0)
    if args.model_kind == 'ann':
        net, metadata = load_official_dimp18(args.upstream, args.checkpoint)
        net = net.cuda()
    else:
        from src.training_dimp import model_from_checkpoint
        net, metadata = model_from_checkpoint(args.checkpoint, device='cuda:0')
    if args.direct_snn:
        if args.model_kind != 'trained' or metadata.get('method') != 'source':
            raise ValueError('Direct ECL evaluation requires a QCFS source checkpoint')
        from src.tracking_dimp import DimpFeatureBackbone
        net.feature_extractor = DimpFeatureBackbone(net.feature_extractor.source,
                                                    T=int(metadata['T']), method='hard').cuda()
        metadata = dict(metadata, evaluation_override='hard_ECL_direct_conversion')
    net.eval().requires_grad_(False)
    tracks = [json.loads(line) for line in args.tracks.read_text().splitlines() if line]
    expected_total = sum(sum(bool(frame['score']) for frame in track['frames']) for track in tracks)
    expected_frames = sum(len(track['frames']) for track in tracks)
    if not tracks or expected_total == 0:
        raise ValueError('A nonempty fixed scored manifest is required')
    rows, details, calls = [], [], 0
    begin = time.time()
    for index, track in enumerate(tracks):
        seed = 2026100321+index
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        params = parameters()
        params.image_sample_size = args.image_size
        params.use_augmentation = False
        params.augmentation = {}
        params.augmentation_expansion_factor = 1
        params.random_shift_factor = 0
        params.scale_factors = [1.0]
        params.device = 'cuda:0'
        params.net = NetWithBackbone(net_path=str(args.checkpoint.resolve()), use_gpu=True)
        params.net.net = net
        tracker = DiMP(params)
        tracker.features_initialized = True
        count = [0]
        def observe(_module, _inputs, _outputs):
            count[0] += 1
        hook = net.feature_extractor.register_forward_hook(observe)
        track_rows = []
        try:
            for offset, frame in enumerate(track['frames']):
                image = cv2.imread(frame['image'])
                if image is None:
                    raise ValueError('Cannot read '+frame['image'])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if offset == 0:
                    tracker.initialize(image, {'init_bbox':track['init']['bbox_xywh']})
                    prediction = track['init']['bbox_xywh']
                    if frame['score']:
                        raise ValueError('Initialization must not be scored')
                else:
                    # No annotation, ID, visibility, mask or GT crop passed here.
                    prediction = tracker.track(image).get('target_bbox')
                row = dict(track_key=track['track_key'], frame=frame['frame'],
                           prediction=prediction, score=bool(frame['score']))
                if row['score']:
                    iou, distance = overlap_and_distance(prediction, frame['gt_bbox_xywh'])
                    row.update(iou=iou, center_error=distance if np.isfinite(distance) else None)
                track_rows.append(row)
                if offset % 10 == 0:
                    write(args.output/'status.json', dict(status='evaluating',
                        track_index=index, tracks=len(tracks), frame=frame['frame'],
                        completed_frame_visits=len(rows)+len(track_rows)))
        finally:
            hook.remove()
        if count[0] != len(track_rows):
            raise RuntimeError('Expected exactly one backbone call per real frame')
        calls += count[0]
        rows.extend(track_rows)
        score_rows = [dict(row, center_error=float('inf') if row.get('center_error') is None
                           else row['center_error']) if row['score'] else row for row in track_rows]
        details.append(dict(track_key=track['track_key'], frames=len(track_rows),
                            backbone_calls=count[0], **sot_metrics(score_rows)))
        write(args.output/f'track_{index:04d}.json', dict(status='complete', rows=track_rows,
                                                         metrics=details[-1]))
    score_rows = [dict(row, center_error=float('inf') if row.get('center_error') is None
                       else row['center_error']) if row['score'] else row for row in rows]
    result = dict(status='complete', **sot_metrics(score_rows), frames=len(rows),
        backbone_calls=calls, tracks=len(tracks), per_track=details, seed=2026100321,
        checkpoint=str(args.checkpoint.resolve()), checkpoint_metadata=metadata,
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(args.tracks.read_bytes()).hexdigest(),
        official_split=False, scope='Fixed local MOT16 GT-initialized SOT manifest; not official SOT results',
        image_size=args.image_size, single_scale=True, initial_augmentation=False,
        direct_conversion=args.direct_snn,
        no_GT_after_initialization=True, seconds=time.time()-begin,
        physical_GPU=os.environ['CUDA_VISIBLE_DEVICES'],
        peak_reserved_MiB=torch.cuda.max_memory_reserved()/1024**2)
    if result['total'] != expected_total or result['frames'] != expected_frames or calls != expected_frames:
        raise RuntimeError('Incomplete SOT evaluation')
    write(args.output/'metrics.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
