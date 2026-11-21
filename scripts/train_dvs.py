"""Shared DVS ResNet sources and matched SNN fine-tuning."""
from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import fcntl
import hashlib
import json
import math
import os
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from src.dvs_classification import DVSFrames, make_ann, install_qcfs, EventResNet
from src.conversion.activation_sites import activation_sites
from src.conversion.channel_qcfs_source import project_thresholds

SEED = 2026100321


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def save(path, state):
    tmp = path.with_suffix('.tmp')
    torch.save(state, tmp)
    tmp.replace(path)


def completed_snn_case(args):
    """Let a later queue invocation adopt a completed independently run arm."""
    if args.stage != 'snn' or args.method is None or args.source is None:
        return False
    summary_path = args.output / args.method / 'summary.json'
    if not summary_path.exists():
        return False
    config = json.loads((args.output / 'config.json').read_text())
    expected = dict(architecture=args.architecture, method=args.method, L=args.L, T=args.T,
                    snn_epochs=args.snn_epochs, batch_size=args.batch_size, seed=SEED,
                    manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                    source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest())
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError('Completed SNN case differs from requested experiment')
    summary = json.loads(summary_path.read_text())
    manifest = json.loads(args.manifest.read_text())
    train_count = sum(row['split'] == 'train' for row in manifest['samples'])
    test_count = sum(row['split'] == 'test' for row in manifest['samples'])
    if (summary.get('status') != 'complete' or summary.get('epochs') != args.snn_epochs
            or summary.get('training_examples') != train_count * args.snn_epochs
            or summary.get('test', {}).get('total') != test_count
            or not (args.output / args.method / 'last.pt').is_file()):
        raise ValueError('SNN completion record is incomplete')
    print(json.dumps(dict(status='adopted_completed_snn_case', output=str(args.output),
                          actual_physical_gpu=config.get('cuda_visible_devices'))), flush=True)
    return True


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_relu_reference(model, args, config):
    """Reuse a completed, unquantized source before fitting a different L."""
    path = args.relu_source
    source_config = json.loads((path.parent.parent/'config.json').read_text())
    for key in ('architecture', 'manifest_sha256'):
        if source_config[key] != config[key]:
            raise ValueError('ReLU reference architecture/data differ')
    if (source_config.get('dataset', 'CIFAR10-DVS') != config['dataset']
            or source_config.get('num_classes', 10) != config['num_classes']):
        raise ValueError('ReLU reference dataset/classes differ')
    state = torch.load(path, map_location='cpu', weights_only=False)
    if (state['epoch'] != args.relu_epochs or state['epochs'] != args.relu_epochs
            or state['seed'] != SEED):
        raise ValueError('Expected the completed fixed-seed ReLU reference')
    model.load_state_dict(state['model'], strict=True)
    return dict(checkpoint=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                epoch=state['epoch'], original_training_examples=state.get('examples'),
                reused_unquantized_reference=True, new_ReLU_training_examples=0,
                note='ReLU weights have no L; thresholds and QAT are fitted anew for requested L')


def loader(dataset, batch, workers, epoch=0, training=False):
    return DataLoader(dataset, batch_size=batch, shuffle=training,
                      num_workers=workers, pin_memory=True,
                      generator=torch.Generator().manual_seed(SEED + epoch),
                      drop_last=False)


@torch.no_grad()
def evaluate(model, dataset, args, event_model=False):
    model.eval()
    correct = total = 0
    loss = 0.
    for currents, target in loader(dataset, args.batch_size, args.workers):
        currents, target = currents.to(args.device), target.to(args.device)
        logits = model(currents if event_model else currents.mean(1))
        loss += F.cross_entropy(logits, target, reduction='sum').item()
        correct += (logits.argmax(1) == target).sum().item()
        total += len(target)
    return dict(correct=correct, total=total, accuracy=100 * correct / total,
                ce=loss / total, split=(dataset.dataset_name+' official test' if dataset.official_split
                                       else 'fixed local 1000-sample held-out test'),
                official_split=dataset.official_split, batch_size=args.batch_size)


def fit(model, train, test, args, output, epochs, lr, event_model=False):
    output.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad),
                                lr=lr, momentum=.9, weight_decay=5e-4)
    start = steps = examples = 0
    last = output / 'last.pt'
    seed_all()
    if last.exists():
        state = torch.load(last, map_location=args.device, weights_only=False)
        if state['epochs'] != epochs or state['batch_size'] != args.batch_size:
            raise ValueError('Resume configuration changed')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start, steps, examples = state['epoch'], state['steps'], state['examples']
        torch.set_rng_state(state['torch_rng'].cpu())
        if args.device.startswith('cuda'):
            torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
    model.to(args.device)
    for epoch in range(start, epochs):
        begin = time.monotonic()
        model.train()
        current_lr = lr * (.01 + .99 * .5 * (1 + math.cos(math.pi * epoch / epochs)))
        if epoch < 5:
            current_lr *= (epoch + 1) / 5
        for group in optimizer.param_groups:
            group['lr'] = current_lr
        count = correct = 0
        sum_loss = 0.
        for currents, target in loader(train, args.batch_size, args.workers, epoch, True):
            currents, target = currents.to(args.device), target.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(currents if event_model else currents.mean(1))
            loss = F.cross_entropy(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            if not event_model:
                project_thresholds(model)
            n = len(target)
            sum_loss += loss.item() * n
            correct += (logits.argmax(1) == target).sum().item()
            count += n
            steps += 1
            examples += n
        row = dict(epoch=epoch+1, epochs=epochs, lr=current_lr, steps=steps,
                   training_examples=examples, train_accuracy=100*correct/count,
                   train_ce=sum_loss/count, last_grad_norm=float(norm),
                   epoch_seconds=time.monotonic()-begin,
                   memory_peak_reserved=(torch.cuda.max_memory_reserved() if args.device.startswith('cuda') else 0))
        if (epoch+1) % 20 == 0 or epoch+1 == epochs:
            row['test'] = evaluate(model, test, args, event_model)
            write_json(output / f'test_epoch{epoch+1}.json', row['test'])
        print(json.dumps(dict(stage=str(output), **row)), flush=True)
        with (output / 'epochs.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                     epoch=epoch+1, epochs=epochs, steps=steps, examples=examples,
                     batch_size=args.batch_size, seed=SEED,
                     torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if args.device.startswith('cuda') else [],
                     python_rng=random.getstate(), numpy_rng=np.random.get_state())
        save(last, state)
        write_json(output / 'status.json', dict(status='training', **row))
    if not (output / f'test_epoch{epochs}.json').exists():
        write_json(output / f'test_epoch{epochs}.json', evaluate(model, test, args, event_model))
    write_json(output / 'summary.json', dict(status='complete', epochs=epochs,
                steps=steps, training_examples=examples,
                test=json.loads((output / f'test_epoch{epochs}.json').read_text())))
    return model


@torch.no_grad()
def calibrate(model, dataset, args):
    samples = {name: [] for name, _ in activation_sites(model)}
    hooks = []
    for name in samples:
        def capture(module, inputs, name=name):
            x = inputs[0].detach().relu().flatten()
            samples[name].append(x[::max(1, x.numel()//8192)].cpu())
        hooks.append(model.get_submodule(name).register_forward_pre_hook(capture))
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(SEED))[:1024]
    model.eval()
    try:
        for currents, _ in loader(Subset(dataset, indices.tolist()), args.batch_size, args.workers):
            model(currents.mean(1).to(args.device))
    finally:
        for hook in hooks:
            hook.remove()
    thresholds = {}
    for name, parts in samples.items():
        values = torch.cat(parts).to(args.device)
        step = ((values.mean()+3*values.std())/args.L).clamp_min(1e-6)
        for _ in range(32):
            quant = (values/step+.5).floor().clamp(0, args.L)
            step = ((values*quant).sum()/quant.square().sum().clamp_min(1)).clamp_min(1e-6)
        thresholds[name] = (args.L*step).cpu().reshape(1)
    return thresholds


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--architecture', choices=['resnet18', 'resnet20'], required=True)
    parser.add_argument('--stage', choices=['source', 'snn'], required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--relu-source', type=Path)
    parser.add_argument('--method', choices=['triangular', 'density'])
    parser.add_argument('--L', type=int, default=4)
    parser.add_argument('--T', type=int, default=4)
    parser.add_argument('--relu-epochs', type=int, default=300)
    parser.add_argument('--qat-epochs', type=int, default=200)
    parser.add_argument('--snn-epochs', type=int, default=160)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--memory-cap-gib', type=float, default=6.)
    args = parser.parse_args()
    if args.relu_source is not None and args.stage != 'source':
        parser.error('--relu-source is only valid for source training')
    args.output.mkdir(parents=True, exist_ok=True)
    output_lock = (args.output / '.training.lock').open('w')
    # A second CPU queue waits for the case owner; it never duplicates GPU work.
    fcntl.flock(output_lock, fcntl.LOCK_EX)
    if completed_snn_case(args):
        return
    torch.set_num_threads(4)
    if args.L < 1 or args.T < 1 or min(args.relu_epochs, args.qat_epochs, args.snn_epochs, args.batch_size) < 1:
        parser.error('L, T, epochs and batch size must be positive')
    if args.device.startswith('cuda'):
        if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('0', '2', '3'):
            parser.error('Set exactly one authorized physical GPU (0, 2 or 3)')
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(args.memory_cap_gib*1024**3/total)
    seed_all()
    print('Loading pooled event counts into CPU RAM.', flush=True)
    train = DVSFrames(args.manifest, 'train', training=True).preload()
    calibration_train = DVSFrames(args.manifest, 'train')
    calibration_train.frames_cache = train.frames_cache
    test = DVSFrames(args.manifest, 'test').preload()
    if train.T != args.T:
        parser.error('Manifest T differs from requested SNN T')
    args.output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(seed=SEED, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                  dataset=train.dataset_name,num_classes=train.num_classes,
                  test_is_official=test.official_split, horizontal_flip=train.horizontal_flip,
                  spatial_size=48, loss='cross entropy',
                  bn_running_policy='fixed source running statistics in SNN arms')
    if args.stage == 'snn':
        if args.source is None or args.method is None:
            parser.error('SNN stage requires --source and --method')
        source_config = json.loads((args.source.parent.parent / 'config.json').read_text())
        if any(source_config[k] != config[k] for k in ('architecture', 'L', 'T', 'manifest_sha256')):
            raise ValueError('Source ANN architecture/L/T/data do not match')
        if (source_config.get('dataset','CIFAR10-DVS')!=config['dataset']
                or source_config.get('num_classes',10)!=config['num_classes']):
            raise ValueError('Source dataset or class count differs')
        config['source_sha256'] = hashlib.sha256(args.source.read_bytes()).hexdigest()
    config_path = args.output / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Existing experiment configuration differs')
    write_json(config_path, config)
    model = make_ann(args.architecture,num_classes=train.num_classes).to(args.device)
    if args.stage == 'source':
        if args.relu_source is None:
            model = fit(model, train, test, args, args.output / 'relu', args.relu_epochs, .1)
        else:
            reference = load_relu_reference(model, args, config)
            write_json(args.output/'relu_reference.json', reference)
        thresholds_path = args.output / 'thresholds.pt'
        if thresholds_path.exists():
            thresholds = torch.load(thresholds_path, map_location='cpu', weights_only=True)
        else:
            thresholds = calibrate(model, calibration_train, args)
            save(thresholds_path, thresholds)
        model = install_qcfs(model, args.L, thresholds)
        fit(model, train, test, args, args.output / 'qat', args.qat_epochs, .01)
        model.eval()
        # Avoid retaining training-sized cached activations during T-bin evaluation.
        evaluation_batch = args.batch_size
        args.batch_size = min(args.batch_size, 64)
        direct = dict(ann=evaluate(model, test, args),
                      qcfs=evaluate(EventResNet(model, args.T, 'qcfs').to(args.device), test, args, True),
                      ecl=evaluate(EventResNet(model, args.T, 'ecl').to(args.device), test, args, True))
        args.batch_size = evaluation_batch
        write_json(args.output / 'direct_conversion.json', direct)
    else:
        if args.source is None or args.method is None:
            parser.error('SNN stage requires --source and --method')
        model = install_qcfs(model, args.L)
        state = torch.load(args.source, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'])
        write_json(args.output / 'direct_conversion.json', dict(
            ann=evaluate(model, test, args),
            qcfs=evaluate(EventResNet(model, args.T, 'qcfs').to(args.device), test, args, True),
            ecl=evaluate(EventResNet(model, args.T, 'ecl').to(args.device), test, args, True),
            source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest()))
        del state
        converted = EventResNet(model, args.T, args.method).to(args.device)
        fit(converted, train, test, args, args.output / args.method, args.snn_epochs, .01, True)


if __name__ == '__main__':
    main()
