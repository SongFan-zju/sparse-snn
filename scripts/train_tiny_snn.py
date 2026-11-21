"""Matched Tiny-ImageNet signed-ECL surrogate finetuning from fixed sources."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from scripts.train_tiny_imagenet import evaluate, load_source, restore_rng, rng_state, write
from src.tiny_imagenet import SEED, TinyECL, TinyImages, sha256
from src.conversion.qcfs_ecl_resnet18 import DualThresholdSequence


def tensor_digest(items):
    digest = hashlib.sha256()
    for name, value in items:
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--method', choices=('density', 'triangular'), required=True)
    parser.add_argument('--L', type=int, choices=(2, 4), required=True)
    parser.add_argument('--T', type=int, choices=(2, 4), required=True)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--memory-cap-gib', type=float, default=44)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--lr', type=float, default=.01)
    parser.add_argument('--eval-every', type=int, default=10)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--profile-only', action='store_true')
    args = parser.parse_args()
    physical = os.environ.get('CUDA_VISIBLE_DEVICES')
    if physical not in ('0', '2', '3'):
        parser.error('Expose exactly one authorized physical GPU0,2,or3')
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (args.output / 'config.json').exists() and not args.resume:
        parser.error('Existing run requires explicit resume')
    torch.set_num_threads(2)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    free, total = torch.cuda.mem_get_info()
    cap = int(args.memory_cap_gib * 1024**3)
    if cap > free - 512 * 1024**2:
        raise RuntimeError('Insufficient free GPU memory for the recorded cap')
    torch.cuda.set_per_process_memory_fraction(cap / total)
    device = torch.device('cuda:0')
    manifest = json.loads(args.manifest.read_text())
    assert manifest['training_images'] == 100000 and manifest['validation_images'] == 10000
    ann, source = load_source(args.source)
    assert source['stage'] == 'qcfs' and source['L'] == args.L
    assert source['manifest_sha256'] == manifest['manifest_sha256']
    assert source['progress']['completed_epochs'] == 60 and source['progress']['training_images'] == 6000000
    architecture = source['architecture']
    model = TinyECL(ann, args.T, args.method).to(device).requires_grad_(True)
    sites = sum(isinstance(m, DualThresholdSequence) for m in model.modules())
    assert sites == (33 if architecture == 'resnet34' else 15)
    initial_state_sha = tensor_digest(model.state_dict().items())
    frozen = {name: value.detach().clone() for name, value in model.named_buffers()}
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=.9, weight_decay=5e-4)
    progress = dict(completed_epochs=0, within_epoch_images=0, training_images=0, updates=0)
    epoch_accumulator = dict(images=0, loss_sum=0., correct=0)
    config = dict(architecture=architecture, method=args.method, L=args.L, T=args.T, seed=SEED,
        epochs=args.epochs, training_images_per_epoch=100000, planned_training_images=args.epochs*100000,
        source=str(args.source.resolve()), source_sha256=sha256(args.source),
        manifest_sha256=manifest['manifest_sha256'], batch_size=args.batch_size,
        physical_gpu=physical, precision='FP32, TF32 disabled', lr=args.lr, min_lr=.0001,
        momentum=.9, weight_decay=.0005, warmup_epochs=5, gradient_clip=5.,
        BN_statistics='frozen', thresholds='fixed source buffers', loss='cross_entropy',
        teacher=None, initial_model_state_sha256=initial_state_sha, activation_sites=sites,
        code_sha256={str(path.relative_to(ROOT)): sha256(path) for path in (
            Path(__file__).resolve(), ROOT/'src/tiny_imagenet.py',
            ROOT/'src/conversion/ecl_triangular_surrogate.py',
            ROOT/'src/conversion/triangular_density_horizon.py',
            ROOT/'src/conversion/triangular_density_triton.py')})
    if args.resume:
        old = torch.load(args.resume, map_location='cpu', weights_only=False)
        for key in ('architecture', 'method', 'L', 'T', 'seed', 'epochs', 'source_sha256', 'manifest_sha256'):
            if old['config'][key] != config[key]:
                raise ValueError('Resume recipe/provenance changed: '+key)
        model.load_state_dict(old['model'], strict=True)
        optimizer.load_state_dict(old['optimizer'])
        progress = old['progress']; epoch_accumulator = old['epoch_accumulator']
        restore_rng(old['rng'])
        with (args.output/'resume_events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(time=time.time(), batch_size=args.batch_size,
                physical_gpu=physical, checkpoint=str(args.resume), progress=progress))+'\n')
    else:
        write(args.output/'config.json', config)
    runtime = dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19],
        physical_gpu=physical, batch_size=args.batch_size, started_at=time.time(), command=sys.argv)
    write(args.output/'runtime.json', runtime)
    stop = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))

    def status(state, **fields):
        write(args.output/'status.json', dict(status=state, state=state, pid=os.getpid(),
            physical_gpu=physical, batch_size=args.batch_size, architecture=architecture,
            method=args.method, L=args.L, T=args.T, epochs=args.epochs,
            epoch=progress['completed_epochs']+int(progress['within_epoch_images']>0),
            updated_at=time.time(), **progress, **fields))

    def buffers_unchanged():
        return all(torch.equal(value, frozen[name]) for name, value in model.named_buffers())

    def save(name):
        if not buffers_unchanged():
            raise RuntimeError('Source BN or threshold buffers changed')
        path = args.output/'checkpoints'/name
        path.parent.mkdir(exist_ok=True)
        state = dict(format='tiny_imagenet_snn_v1', model=model.state_dict(),
            optimizer=optimizer.state_dict(), config=config, progress=copy.deepcopy(progress),
            epoch_accumulator=copy.deepcopy(epoch_accumulator), rng=rng_state())
        temporary = path.with_suffix('.tmp')
        torch.save(state, temporary); temporary.replace(path)
        return path

    train = TinyImages(manifest, 'train', training=True)
    validation = DataLoader(TinyImages(manifest, 'validation'), batch_size=64,
        num_workers=args.workers, pin_memory=True)
    started = time.time(); invocation_updates = 0; first_gradient = None
    measured_images = 0; measured_seconds = 0.; last_checkpoint = None
    try:
        while progress['completed_epochs'] < args.epochs:
            epoch = progress['completed_epochs']+1
            train.epoch = epoch
            order = torch.randperm(100000, generator=torch.Generator().manual_seed(SEED+epoch)).tolist()
            remaining = order[progress['within_epoch_images']:]
            loader = DataLoader(train, batch_size=args.batch_size, sampler=remaining,
                num_workers=args.workers, pin_memory=True, drop_last=False,
                **({'prefetch_factor':2} if args.workers else {}))
            cosine = .5*(1+math.cos(math.pi*max(0,epoch-6)/max(1,args.epochs-6)))
            lr = (.0001+(args.lr-.0001)*cosine)*min(1.,epoch/5)
            for group in optimizer.param_groups: group['lr'] = lr
            model.train(); epoch_started = time.time()
            for images, labels in loader:
                images = images.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
                parity = None
                if args.profile_only and invocation_updates == 0:
                    model.eval()
                    actual = model(images[:2]).detach()
                    reference = TinyECL(ann, args.T, 'ecl').to(device)
                    with torch.no_grad(): expected = reference(images[:2])
                    parity = bool(torch.equal(actual, expected))
                    del reference, actual, expected
                    if not parity: raise RuntimeError('Gradient-enabled hard forward differs from ECL')
                    model.train()
                tick = time.time(); optimizer.zero_grad(set_to_none=True)
                logits = model(images); loss = nn.functional.cross_entropy(logits, labels)
                if not bool(torch.isfinite(loss)): raise RuntimeError('Nonfinite SNN training loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                if invocation_updates == 0:
                    missing = [n for n,p in model.named_parameters() if p.requires_grad and p.grad is None]
                    if missing: raise RuntimeError('Missing trainable gradients: '+str(missing))
                    first_gradient = dict(all_trainable_gradients_present_finite=True,
                        active_parameter_tensors=sum(p.grad is not None for p in model.parameters()),
                        first_conv_grad_l1=float(next(model.parameters()).grad.abs().sum()),
                        gradient_norm=float(norm))
                torch.cuda.synchronize()
                if args.profile_only:
                    report = dict(status='complete', architecture=architecture, method=args.method,
                        L=args.L, T=args.T, batch_size=len(images), physical_gpu=physical,
                        seconds=time.time()-tick, loss=float(loss), gradient_check=first_gradient,
                        hard_forward_equal_ECL=parity, frozen_buffers_unchanged=buffers_unchanged(),
                        peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
                        peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20, optimizer_updates=0)
                    write(args.output/'profile.json', report); print(json.dumps(report), flush=True)
                    return
                optimizer.step(); torch.cuda.synchronize()
                elapsed = time.time()-tick; count = len(images)
                progress['within_epoch_images'] += count; progress['training_images'] += count
                progress['updates'] += 1; invocation_updates += 1
                epoch_accumulator['images'] += count
                epoch_accumulator['loss_sum'] += float(loss)*count
                epoch_accumulator['correct'] += int((logits.argmax(1)==labels).sum())
                if invocation_updates > 5:
                    measured_images += count; measured_seconds += elapsed
                if invocation_updates == 1 or invocation_updates % 10 == 0:
                    rate = measured_images/measured_seconds if measured_seconds else count/max(elapsed,1e-9)
                    status('running', last_loss=float(loss), images_per_second=rate,
                        estimated_train_epoch_seconds=100000/rate,
                        peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
                        peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20)
                if stop[0] or progress['within_epoch_images'] == 100000: break
            if progress['within_epoch_images'] == 100000:
                assert epoch_accumulator['images'] == 100000
                progress['completed_epochs'] += 1; progress['within_epoch_images'] = 0
                row = dict(epoch=epoch, train_loss=epoch_accumulator['loss_sum']/100000,
                    train_accuracy=epoch_accumulator['correct']/1000, lr=lr,
                    epoch_seconds=time.time()-epoch_started,
                    pair_order_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest(), **progress)
                epoch_accumulator = dict(images=0, loss_sum=0., correct=0)
                last_checkpoint = save('last.pt')
                if epoch % 20 == 0 or epoch == args.epochs: save(f'epoch{epoch}.pt')
                if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
                    row['validation'] = evaluate(model, validation, device, args.output/f'test_epoch{epoch}.json')
                with (args.output/'epochs.jsonl').open('a') as stream:
                    stream.write(json.dumps(row)+'\n')
                print(json.dumps(row), flush=True)
            if stop[0]: break
        if last_checkpoint is None or progress['within_epoch_images']:
            last_checkpoint = save('last.pt')
        complete = progress['completed_epochs'] == args.epochs
        summary = dict(status='complete' if complete else 'interrupted', architecture=architecture,
            method=args.method, L=args.L, T=args.T, seed=SEED, epochs=args.epochs, **progress,
            checkpoint=str(last_checkpoint.resolve()), checkpoint_sha256=sha256(last_checkpoint),
            source_sha256=config['source_sha256'], initial_model_state_sha256=initial_state_sha,
            frozen_buffers_unchanged=buffers_unchanged(), first_gradient_check=first_gradient,
            physical_gpu=physical, batch_size=args.batch_size, elapsed_seconds=time.time()-started)
        if complete: summary['test'] = json.loads((args.output/f'test_epoch{args.epochs}.json').read_text())
        write(args.output/'summary.json', summary); status(summary['status'])
        print(json.dumps(summary), flush=True)
    except BaseException as error:
        status('error', error=repr(error)); raise


if __name__ == '__main__':
    main()
