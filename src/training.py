"""One training loop for ANN preparation, QCFS QAT and hard-SNN adaptation."""
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from src.calibration import calibrate
from src.conversion.channel_qcfs_source import project_thresholds, threshold_parameters
from src.data import load_cifar, mixed_batch, partitions
from src.evaluation import evaluate
from src.models.factory import backbone, convert, install_qat, load_ann, load_model, load_qat
from src.runtime import cpu_state, digest, generators, report, restore_rng, rng_state, save


def optimizer_for(model, cfg):
    thresholds = threshold_parameters(model)
    ids = {id(p) for p in thresholds}
    groups = [dict(params=[p for p in model.parameters() if p.requires_grad and id(p) not in ids],
                   name='source', lr_multiplier=1., weight_decay=cfg['weight_decay'])]
    if thresholds:
        groups.append(dict(params=thresholds, name='threshold', lr_multiplier=.1, weight_decay=0.))
    return torch.optim.SGD(groups, lr=cfg['lr'], momentum=cfg['momentum'])


def schedule(optimizer, cfg, epoch):
    warmup, peak, minimum = cfg['warmup_epochs'], cfg['lr'], cfg['min_lr']
    lr = peak * epoch / warmup if epoch <= warmup else minimum + (peak - minimum) * .5 * (
        1 + math.cos(math.pi * (epoch - warmup) / (cfg['epochs'] - warmup)))
    for group in optimizer.param_groups:
        group['lr'] = lr * group['lr_multiplier']
    return lr


def train_step(runner, model, optimizer, images, y1, y2, lam, cfg, teacher=None):
    optimizer.zero_grad(set_to_none=True)
    probabilities = None
    if cfg['kd_weight']:
        if teacher is None or teacher.training:
            raise ValueError('KD requires a frozen teacher in eval mode')
        with torch.no_grad():
            probabilities = torch.cat([(teacher(x) / cfg['temperature']).softmax(1)
                                       for x in images.split(cfg['teacher_chunk'])])
    logits = runner(images)
    ce = lam * F.cross_entropy(logits, y1) + (1 - lam) * F.cross_entropy(logits, y2)
    kd = logits.new_zeros(()) if probabilities is None else cfg['temperature']**2 * F.kl_div(
        (logits / cfg['temperature']).log_softmax(1), probabilities, reduction='batchmean')
    loss = ce + cfg['kd_weight'] * kd
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite training loss')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'],
                                        error_if_nonfinite=True) if cfg['gradient_clip'] else 0.
    optimizer.step()
    if cfg['stage'] == 'qat':
        project_thresholds(model)
    return dict(loss=float(loss.detach()), ce=float(ce.detach()), kd=float(kd.detach()),
                gradient_norm=float(norm))


def train_epoch(model, runner, optimizer, teacher, raw, labels, indices, streams, cfg, device):
    model.train()
    order = indices[torch.randperm(len(indices), generator=streams['order']).numpy()]
    totals = dict(loss=0., ce=0., kd=0.)
    max_norm, steps = 0., 0
    for offset in range(0, len(order), cfg['batch_size']):
        chosen = order[offset:offset + cfg['batch_size']]
        images, y1, y2, lam, _ = mixed_batch(raw, labels, chosen, streams, device, cfg['dataset'])
        observation = train_step(runner, model, optimizer, images, y1, y2, lam, cfg, teacher)
        for name in totals:
            totals[name] += observation[name] * len(chosen)
        max_norm = max(max_norm, observation['gradient_norm'])
        steps += 1
    return dict(**{name: value / len(order) for name, value in totals.items()},
                gradient_norm=max_norm, steps=steps, training_examples=len(order))


def prepare_model(cfg, args, raw, roles, device):
    if cfg['stage'] == 'ann':
        return load_ann(args.source, cfg) if args.source else backbone(cfg)
    if not args.source:
        raise ValueError('QAT requires --source ANN; SNN adaptation requires --source QAT')
    if cfg['stage'] == 'snn':
        return convert(load_qat(args.source, cfg), cfg, training=True)
    source = load_ann(args.source, cfg)
    if args.thresholds:
        saved = torch.load(args.thresholds, map_location='cpu', weights_only=False)
        if 'source_sha256' in saved and saved['source_sha256'] != digest(args.source):
            raise ValueError('Thresholds were calibrated from a different source checkpoint')
        thresholds = saved.get('thresholds', saved)
    else:
        if cfg.get('require_thresholds'):
            raise ValueError('This paper QAT recipe requires the supplied --thresholds initialization')
        # C10's reported calibration was on CPU; R50's was on the training GPU.
        calibration_device = torch.device('cpu') if cfg['dataset'] == 'cifar10' else device
        thresholds = calibrate(source, raw, roles['calibration'], cfg, calibration_device)
    return install_qat(source, thresholds, cfg)


def train(cfg, args, device):
    if cfg['epochs'] < 1:
        raise ValueError('This preset describes an external ANN; use it for evaluation, not retraining')
    if cfg['kd_weight'] and not args.teacher:
        raise ValueError('This recipe requires --teacher (an ordinary ANN checkpoint)')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(out.iterdir()):
        raise FileExistsError('Choose an empty output directory, or use --resume')
    saved = None
    if args.resume:
        saved = torch.load(out / 'latest.pt', map_location='cpu', weights_only=False)
        if saved.get('format_version') != 1:
            raise ValueError('Resume requires a checkpoint from this public runner')
        # Batch changes are allowed; the images and epoch budget remain fixed.
        if {k: v for k, v in saved['config'].items() if k != 'batch_size'} != {
                k: v for k, v in cfg.items() if k != 'batch_size'}:
            raise ValueError('Resume configuration differs from the saved experiment')
    raw, labels = load_cifar(args.data, cfg['dataset'], 'train')
    roles = partitions(cfg['dataset'], len(raw), args.split_file, cfg['seed'])
    if len(roles['training']) != cfg['training_examples']:
        raise ValueError('Training image count differs from the declared recipe')
    model = (load_model(out / 'latest.pt', cfg, training=True) if saved else
             prepare_model(cfg, args, raw, roles, device)).to(device)
    teacher = load_ann(args.teacher, cfg).to(device).eval().requires_grad_(False) if cfg['kd_weight'] else None
    if args.data_parallel:
        if device.type != 'cuda' or device.index != 0 or torch.cuda.device_count() < 2:
            raise ValueError('--data-parallel requires cuda:0 and at least two visible GPUs')
        runner = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    else:
        runner = model
    optimizer = optimizer_for(model, cfg)
    streams = generators(cfg['seed'])
    inputs = dict(source_sha256=digest(args.source) if args.source else None,
                  teacher_sha256=digest(args.teacher) if args.teacher else None,
                  thresholds_sha256=digest(args.thresholds) if args.thresholds else None,
                  split_sha256=digest(args.split_file) if cfg['dataset'] == 'cifar100' else None)
    first, steps, elapsed = 1, 0, 0.
    if saved:
        if inputs['teacher_sha256'] != saved['inputs']['teacher_sha256'] or inputs['split_sha256'] != saved['inputs']['split_sha256']:
            raise ValueError('Resume teacher or data partition changed')
        inputs = saved['inputs']
        optimizer.load_state_dict(saved['optimizer'])
        restore_rng(saved['rng'], streams)
        first, steps, elapsed = saved['epoch'] + 1, saved['steps'], saved['training_seconds']
        if first > cfg['epochs']:
            report(dict(status='already_complete', epoch=saved['epoch'], output=str(out)))
            return
    runtime = dict(device=str(device), data_parallel=args.data_parallel,
                   visible_gpus=torch.cuda.device_count() if device.type == 'cuda' else 0,
                   torch_version=str(torch.__version__), training_examples=len(roles['training']))
    save(out / 'config.json', dict(recipe=cfg, inputs=inputs, runtime=runtime), json_format=True)
    started = time.monotonic()
    for epoch in range(first, cfg['epochs'] + 1):
        epoch_started = time.monotonic()
        lr = schedule(optimizer, cfg, epoch)
        row = train_epoch(model, runner, optimizer, teacher, raw, labels, roles['training'], streams, cfg, device)
        steps += row.pop('steps')
        row.update(epoch=epoch, steps=steps, lr=lr, batch_size=cfg['batch_size'],
                   epoch_seconds=time.monotonic() - epoch_started,
                   training_seconds=elapsed + time.monotonic() - started)
        if epoch in cfg['eval_epochs'] and cfg['dataset'] == 'cifar100':
            names = ['selection'] + (['confirmation'] if epoch == cfg['epochs'] else [])
            row['evaluation'] = {name: evaluate(model, raw, labels, roles[name], cfg['dataset'], device, 1024)
                                 for name in names}
        if epoch == 1 or epoch % args.checkpoint_every == 0 or epoch == cfg['epochs'] or epoch in cfg['eval_epochs']:
            checkpoint = dict(format_version=1, config=cfg, inputs=inputs, runtime=runtime,
                              model=cpu_state(model), optimizer=optimizer.state_dict(), rng=rng_state(streams),
                              epoch=epoch, steps=steps, training_seconds=row['training_seconds'])
            save(out / 'latest.pt', checkpoint)
            if epoch in cfg['eval_epochs'] or epoch == cfg['epochs']:
                save(out / f'epoch{epoch}.pt', checkpoint)
        with (out / 'epochs.jsonl').open('a') as handle:
            handle.write(json.dumps(row, allow_nan=False) + '\n')
        report(row)
    report(dict(status='complete', config=cfg, final=row), out / 'summary.json')
