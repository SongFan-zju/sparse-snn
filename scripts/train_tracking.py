#!/usr/bin/env python3
"""ViStream MOT16: clip-consistent training and one streaming trajectory/frame."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F

from src.tracking_vistream import ViStreamBackbone, load_source
from src.tracking_association import reconstruction_similarity
from src.tracking_clip import ClipViStreamBackbone, CONTEXT_RUNTIME, configure_context_runtime
from src.tracking_vistream_io import (MOT16Sequence, roi_features,
    training_identity_map, training_id_targets)

SEED = 2026100321
TRAIN_SEQUENCES = ('MOT16-02', 'MOT16-04', 'MOT16-05', 'MOT16-09')
EVAL_SEQUENCES = ('MOT16-10', 'MOT16-11', 'MOT16-13')


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def make_model(args, device, raw_checkpoint=None):
    source, _ = load_source(args.upstream, args.source, args.L)
    model_class = ClipViStreamBackbone if args.action == 'train' else ViStreamBackbone
    model = model_class(source, args.T, args.method).to(device)
    if args.checkpoint:
        raw = raw_checkpoint if raw_checkpoint is not None else torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(raw['model'], strict=True)
    return model


def train(args, model, device):
    output = Path(args.output)
    clip_frames = 4
    context_frames = 4 if args.continuous_context else 0
    training_state = ('rebuild_context_and_target_in_one_graph' if context_frames
                      else 'reset_per_clip_graph_retained_until_step')
    identity_map = training_identity_map(args.data)
    sequences = [MOT16Sequence(args.data, name, image_size=(544,304), training_targets=True)
                 for name in TRAIN_SEQUENCES]
    frames_per_epoch = sum(len(sequence) for sequence in sequences)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr, weight_decay=1e-4)
    epoch0, updates = 0, 0
    latest = output/'latest.pt'
    if args.resume and latest.exists():
        saved = torch.load(latest, map_location='cpu', weights_only=False)
        recipe = saved.get('config', {})
        if (recipe.get('training_state') != training_state
                or recipe.get('clip_frames') != clip_frames
                or recipe.get('context_frames',0) != context_frames
                or recipe.get('runtime_policy','legacy_default') != args.runtime_policy):
            raise ValueError('Resume requires a clip-training checkpoint; per-frame runs must remain separately labeled')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        epoch0, updates = saved['epoch'], saved['updates']
        random.setstate(saved['random']); np.random.set_state(saved['numpy'])
        torch.set_rng_state(saved['torch']); torch.cuda.set_rng_state_all(saved['cuda'])
    for epoch in range(epoch0+1,args.epochs+1):
        start = time.monotonic()
        losses, frame_count, supervised_frames, maximum_gradient_norm = [], 0, 0, 0.
        context_count = 0
        model.train()
        for sequence in sequences:
            for first in range(0,len(sequence),clip_frames):
                frames = [sequence[i] for i in range(first,min(first+clip_frames,len(sequence)))]
                images = torch.stack([frame.image for frame in frames]).to(device)
                context = [sequence[i].image for i in range(max(0,first-context_frames),first)]
                context_images = torch.stack(context).to(device) if context else images[:0]
                optimizer.zero_grad(set_to_none=True)
                # All frames, including the absolute anchor, stay on the graph.
                # Weights are constant for every tick of this clip.
                features = (model.forward_clip_with_context(context_images,images)
                            if context_frames else model.forward_clip(images))
                context_count += len(context)
                bank, clip_losses = {}, []
                for frame,feature in zip(frames,features):
                    roi,selected = roi_features(feature,frame.observations,image_size=(544,304))
                    targets = training_id_targets(sequence,frame,identity_map).to(device)[selected]
                    if len(roi) and len(bank)>=2:
                        bank_ids = sorted(bank)
                        lookup = {identity:i for i,identity in enumerate(bank_ids)}
                        keep = torch.tensor([int(identity) in lookup for identity in targets],device=device)
                        if bool(keep.any()):
                            memory = torch.stack([bank[identity] for identity in bank_ids])
                            labels = torch.tensor([lookup[int(identity)] for identity in targets[keep]],device=device)
                            logits = reconstruction_similarity(memory,roi).T[keep]/.07
                            frame_loss = F.cross_entropy(logits,labels)
                            if not bool(torch.isfinite(frame_loss)):
                                raise RuntimeError('Non-finite clip association loss')
                            clip_losses.append(frame_loss)
                            supervised_frames += 1
                    # Templates stop gradient; the same frame's state/anchor
                    # graph remains connected to later frame predictions.
                    for identity in targets[targets>=0].unique().tolist():
                        bank[int(identity)] = roi[targets==identity].mean(0).detach()
                if clip_losses:
                    loss = torch.stack(clip_losses).mean()
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                    if not bool(torch.isfinite(norm)):
                        raise RuntimeError('Non-finite clip parameter gradient')
                    maximum_gradient_norm = max(maximum_gradient_norm,float(norm))
                    optimizer.step()
                    updates += 1
                    losses.append(float(loss.detach()))
                # No old-weight cache is carried into the next training clip.
                model.reset()
                # Also release graph references when no usable ID loss exists.
                del features, images, roi, feature, bank, clip_losses, context_images, context
                frame_count += len(frames)
                if (first//clip_frames)%12==0 or first+len(frames)==len(sequence):
                    row = dict(status='training',epoch=epoch,sequence=sequence.name,
                        frame=first+len(frames),sequence_length=len(sequence),updates=updates,
                        frames_this_epoch=frame_count,clip_frames=clip_frames,
                        context_frames=context_frames,context_images_this_epoch=context_count,
                        mean_association_loss=float(np.mean(losses)) if losses else None,
                        seconds=time.monotonic()-start,pid=os.getpid(),physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))
                    write_json(output/'status.json',row);print(json.dumps(row),flush=True)
                if args.smoke_frames and frame_count>=args.smoke_frames:
                    if not updates or maximum_gradient_norm<=0:
                        raise RuntimeError('Clip smoke did not exercise learning')
                    write_json(output/'smoke.json',dict(status='passed',frames=frame_count,updates=updates,
                        supervised_frames=supervised_frames,clip_frames=clip_frames,
                        context_frames=context_frames,context_images=context_count,
                        maximum_gradient_norm=maximum_gradient_norm,
                        mean_loss=float(np.mean(losses)) if losses else None,
                        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))
                    return
        assert frame_count==frames_per_epoch
        row = dict(epoch=epoch,frames=frame_count,updates=updates,supervised_frames=supervised_frames,
                   context_frames=context_frames,context_images=context_count,
                   clip_frames=clip_frames,seconds=time.monotonic()-start,
                   mean_association_loss=float(np.mean(losses)) if losses else None)
        payload = dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,updates=updates,
            config=dict(vars(args),association_objective='author_reconsdot_ID_CE',clip_frames=clip_frames,
                        training_state=training_state,context_frames=context_frames,
                        phase_horizon=(clip_frames+context_frames)*args.T),
            random=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
        torch.save(payload,output/'latest.pt.tmp');(output/'latest.pt.tmp').replace(latest)
        torch.save(payload,output/f'epoch{epoch}.pt')
        with (output/'epochs.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    write_json(output/'training_complete.json',dict(status='complete',epochs=args.epochs,
        updates=updates,train_sequences=TRAIN_SEQUENCES,frames_per_epoch=frames_per_epoch,
        clip_frames=clip_frames,context_frames=context_frames,training_state=training_state,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['train', 'eval'])
    parser.add_argument('--upstream', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--method', choices=['hard', 'triangular', 'density'], default='density')
    parser.add_argument('--L', type=int, default=7)
    parser.add_argument('--T', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke-frames', type=int, default=0)
    parser.add_argument('--continuous-context', action='store_true',
                        help='Use preceding four real frames on the same graph; supervise four target frames')
    parser.add_argument('--memory-cap-gib',type=float)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(exist_ok=True, parents=True)
    previous_config = output/'config.json'
    if previous_config.exists():
        previous = json.loads(previous_config.read_text())
        if previous.get('context_frames',0) != (4 if args.continuous_context else 0):
            raise ValueError('Context revision requires a separate output directory; preserve historical runs')
    random.seed(SEED)
    np.random.seed(SEED % 2**32)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    raw_checkpoint = torch.load(args.checkpoint,map_location='cpu',weights_only=False) if args.checkpoint else None
    recorded_runtime = raw_checkpoint.get('config',{}).get('runtime_policy') if raw_checkpoint else None
    args.runtime_policy = CONTEXT_RUNTIME if args.continuous_context or recorded_runtime==CONTEXT_RUNTIME else 'legacy_default'
    if args.runtime_policy==CONTEXT_RUNTIME:
        device = configure_context_runtime(args.device,SEED)
    if args.memory_cap_gib is not None:
        if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('0','2','3'):
            raise ValueError('Resource-capped jobs require one authorized physical GPU')
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(args.memory_cap_gib*2**30/total,device)
    model = make_model(args,device,raw_checkpoint)
    del raw_checkpoint
    write_json(output/'config.json', dict(**vars(args), seed=SEED,
               association_objective='author_reconsdot_ID_CE', memory_ttl='within_training_clip', clip_frames=4,
               training_state=('rebuild_context_and_target_in_one_graph' if args.continuous_context
                               else 'reset_per_clip_graph_retained_until_step'),
               context_frames=4 if args.continuous_context and args.action=='train' else 0,
               convolution_execution=('native_per_frame_batch' if args.continuous_context or args.action=='eval'
                                      else 'legacy_folded_clip_batch'),
               phase_horizon=(8 if args.continuous_context else 4)*args.T if args.action=='train' else None,
               source_count_limit_L=args.L, original_source_level=16,
               encoding='frame_delta_then_zero', state='carried_across_frames',
               evaluation_split='MOT16 public train heldout, not official test',
               train_sequences=TRAIN_SEQUENCES, eval_sequences=EVAL_SEQUENCES,
               train_image_size=[544,304], eval_image_size=[1088,608],
               fixed_T_no_equilibrium_early_exit=True, trainable_stage='layer3',
               trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
               physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))
    if args.action == 'train':
        train(args, model, device)
    else:
        from src.tracking_vistream_io import evaluate_tracking
        model.eval()
        evaluate_tracking(model, args.data, EVAL_SEQUENCES, output, args.upstream,
                          image_size=(1088,608), device=str(device))


if __name__ == '__main__':
    main()
