"""CIFAR100 adaptation of the author Fast-SNN ResNet18 classification graph."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import copy
import fcntl
import gc
import json
import math
import os
import time

import numpy as np
import torch
from torch.nn import functional as F

from src.classification_fast_snn import author_runtime, make_ann, FastClassifier
from src.data import load_cifar, normalize, augment
from src.runtime import configure, cpu_state, digest, generators, restore_rng, rng_state

SEED = 2026100321


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.pending')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def save(path, value):
    temp = path.with_suffix(path.suffix + '.pending')
    torch.save(value, temp)
    temp.replace(path)


@torch.no_grad()
def evaluate(model, test, labels, batch):
    model.eval()
    old_tf32 = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    correct = count = 0
    ce = 0.
    try:
        for start in range(0, len(test), batch):
            x, y = test[start:start+batch], labels[start:start+batch]
            logits = model(x)
            correct += int((logits.argmax(1) == y).sum())
            count += len(y)
            ce += float(F.cross_entropy(logits, y, reduction='sum'))
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = old_tf32
    return dict(accuracy=100*correct/count, correct=correct, total=count, ce=ce/count,
                dataset='CIFAR100 official test', batch_size=batch)


def fit(model, raw, targets, test, test_labels, args, folder, epochs, lr, wd, snn=False):
    folder.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad),
                                lr=lr, momentum=.9, weight_decay=wd)
    streams = generators(SEED)
    epoch0 = steps = examples = 0
    seconds = 0.
    last = folder / 'latest.pt'
    if last.exists():
        state = torch.load(last, map_location='cpu', weights_only=False)
        if state['epochs'] != epochs or state['batch_size'] != args.batch_size:
            raise ValueError('Resume budget or batch differs')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        epoch0, steps, examples, seconds = state['epoch'], state['steps'], state['examples'], state['seconds']
        restore_rng(state['rng'], streams)
    frozen = {n: v.detach().cpu().clone() for n, v in model.named_buffers()} if snn else None
    for epoch in range(epoch0, epochs):
        begin = time.monotonic()
        model.train()
        torch.backends.cudnn.allow_tf32 = not snn
        torch.backends.cuda.matmul.allow_tf32 = not snn
        if snn:
            rate = lr*(.01+.99*.5*(1+math.cos(math.pi*epoch/epochs)))
            if epoch < 5:
                rate *= (epoch+1)/5
        else:
            rate = lr * (.1 ** (int(epoch >= 150)+int(epoch >= 225)))
        for group in optimizer.param_groups:
            group['lr'] = rate
        order = torch.randperm(len(raw), generator=streams['order']).numpy()
        total = correct = 0
        ce = 0.
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start+args.batch_size]
            images = augment(raw, indices, streams['augment'], args.device, 'cifar100')
            labels = torch.as_tensor(targets[indices], device=args.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('Non-finite classification loss')
            loss.backward()
            if snn:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            else:
                norm = None
            optimizer.step()
            if not snn:
                with torch.no_grad():
                    for name, parameter in model.named_parameters():
                        if name.endswith('.act_alpha'):
                            parameter.clamp_(min=1e-4)
            total += len(indices)
            examples += len(indices)
            steps += 1
            correct += int((logits.argmax(1) == labels).sum())
            ce += float(loss)*len(indices)
        if frozen is not None:
            assert all(torch.equal(v.detach().cpu(), frozen[n]) for n, v in model.named_buffers())
        duration = time.monotonic()-begin
        seconds += duration
        row = dict(epoch=epoch+1, epochs=epochs, steps=steps, examples=examples,
                   lr=rate, train_accuracy=100*correct/total, ce=ce/total,
                   batch_size=args.batch_size, epoch_seconds=duration, seconds=seconds,
                   grad_norm=float(norm) if norm is not None else None,
                   peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2)
        if (epoch+1) % 20 == 0 or epoch+1 == epochs:
            row['test'] = evaluate(model, test, test_labels, args.eval_batch)
            write(folder/f'test_epoch{epoch+1}.json', row['test'])
        save(last, dict(model=cpu_state(model), optimizer=copy.deepcopy(optimizer.state_dict()),
                       epoch=epoch+1, epochs=epochs, steps=steps, examples=examples,
                       seconds=seconds, batch_size=args.batch_size, seed=SEED,
                       rng=rng_state(streams)))
        write(folder/'status.json', dict(status='training', pid=os.getpid(), **row))
        with (folder/'epochs.jsonl').open('a') as log:
            log.write(json.dumps(row, allow_nan=False)+'\n')
        print(json.dumps(dict(folder=str(folder), **row), allow_nan=False), flush=True)
    result = json.loads((folder/f'test_epoch{epochs}.json').read_text())
    write(folder/'summary.json', dict(status='complete', epochs=epochs, steps=steps,
                                     training_examples=examples, seconds=seconds, test=result))
    return model


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--stage', choices=['source', 'snn', 'preflight'], required=True)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--init', type=Path, help='Public CIFAR10 2bit source to adapt; replace only its10-class head')
    parser.add_argument('--method', choices=['triangular', 'density'])
    parser.add_argument('--L', type=int, default=3)
    parser.add_argument('--T', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=160)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--eval-batch', type=int, default=128)
    parser.add_argument('--memory-cap-gib', type=float, default=32.)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('0','2','3') or args.device != 'cuda:0':
        parser.error('Use exactly one authorized physical GPU0/2/3')
    if (args.L, args.T) != (3,3):
        parser.error('Initial Fast-SNN CIFAR100 experiment is fixed L3/T3')
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output/'.training.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    configure(args.device, SEED, 4)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(args.memory_cap_gib*1024**3/total)
    runtime = author_runtime(args.upstream)
    config = {k: str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(seed=SEED, dataset='CIFAR100', training_count=50000, test_count=10000,
                  architecture='author Fast-SNN ResNet18 [3,3,2], widths128/256/512',
                  positive_comparison='>=', final_logits='sum over time',
                  cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
                  source_schedule=('public CIFAR10 2bit -> CIFAR100 2bit300; SGD, steps150/225' if args.init
                                   else 'ReLU300 -> 4bit300 -> 3bit300 -> 2bit300; SGD, steps150/225'),
                  source_batch_lr_policy='Keep author base learning rate; batch recorded',
                  source_training_precision='FP32 parameters with TF32 convolutions; deterministic CUDA; evaluations disable TF32',
                  snn_training_precision='FP32, TF32 disabled for both gradient arms',
                  snn_recipe='CE160 .01 cosine warm5 clip5; fixed source BN statistics and thresholds')
    if args.stage != 'source':
        config['source_schedule'] = 'Shared adapted author source; see source/config.json'
    if args.source:
        config['source_sha256'] = digest(args.source)
    if args.init:
        config['init_sha256'] = digest(args.init)
    existing = args.output/'config.json'
    if existing.exists() and json.loads(existing.read_text()) != config:
        raise ValueError('Existing experiment configuration differs')
    write(existing, config)
    raw, targets = load_cifar(args.data, 'cifar100', 'train')
    test_raw, labels = load_cifar(args.data, 'cifar100', 'test')
    if len(raw) != 50000 or len(test_raw) != 10000 or len(np.unique(targets)) != 100:
        raise ValueError('Expected complete official CIFAR100 train/test')
    test = normalize(test_raw,'cifar100').to(args.device)
    test_labels = torch.as_tensor(labels,device=args.device)
    if args.stage == 'source':
        previous = None
        stages = [(2,.04,3e-5)] if args.init else [(32,.1,5e-4),(4,.04,1e-4),(3,.04,1e-4),(2,.04,3e-5)]
        for bit, rate, decay in stages:
            folder = args.output/f'bit{bit}'
            model = make_ann(runtime, bit).to(args.device)
            if args.init:
                initial = torch.load(args.init, map_location='cpu', weights_only=False)['state_dict']
                head = {'layer4.block.2.weight', 'layer4.block.2.bias'}
                if initial['layer4.block.2.weight'].shape != (10,512):
                    raise ValueError('Expected author public CIFAR10 head')
                loaded = model.load_state_dict({k:v for k,v in initial.items() if k not in head}, strict=False)
                if set(loaded.missing_keys) != head or loaded.unexpected_keys:
                    raise ValueError('Only the public10-class head may be replaced')
                del initial
            if previous is not None:
                loaded = model.load_state_dict(previous, strict=False)
                expected = {n for n,_ in model.named_parameters() if n.endswith('.act_alpha')} if bit == 4 else set()
                if set(loaded.missing_keys) != expected or loaded.unexpected_keys:
                    raise ValueError('Unexpected source-stage state mapping')
            configure(args.device, SEED, 4)
            fit(model, raw, targets, test, test_labels, args, folder, 300, rate, decay)
            previous = cpu_state(model)
            del model
            torch.cuda.empty_cache()
        write(args.output/'summary.json', dict(status='complete', final_source=str((args.output/'bit2/latest.pt').resolve()),
                                               source_sha256=digest(args.output/'bit2/latest.pt')))
    elif args.stage == 'preflight':
        if args.source is None:
            parser.error('Preflight requires the common source')
        model = make_ann(runtime,2).to(args.device)
        state = torch.load(args.source,map_location='cpu',weights_only=False)
        model.load_state_dict(state['model'],strict=True)
        del state
        streams = generators(SEED)
        indices = np.arange(args.batch_size)
        images = augment(raw,indices,streams['augment'],args.device,'cifar100')
        labels = torch.as_tensor(targets[indices],device=args.device)
        native = runtime.model.resnet18(spike=True,bit=2,num_classes=100).to(args.device).eval()
        native.load_state_dict(model.state_dict(),strict=True)
        with torch.no_grad():
            expected = native(images.clone()).cpu()
        del native
        gc.collect()
        torch.cuda.empty_cache()
        records = []
        for method in ['triangular','density']:
            configure(args.device,SEED,4)
            candidate = FastClassifier(model,runtime,args.T,method).to(args.device).train()
            frozen = {n:v.detach().cpu().clone() for n,v in candidate.named_buffers()}
            optimizer = torch.optim.SGD(candidate.parameters(),lr=.01,momentum=.9,weight_decay=5e-4)
            torch.cuda.reset_peak_memory_stats()
            rows = []
            for step in range(2):
                begin = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                logits = candidate(images)
                if step == 0 and not torch.equal(logits.detach().cpu(),expected):
                    raise RuntimeError('Adapter training hard forward differs from native author SNN')
                loss = F.cross_entropy(logits,labels)
                loss.backward()
                gradients = [p.grad for p in candidate.parameters()]
                if any(g is None or not bool(g.isfinite().all()) for g in gradients):
                    raise RuntimeError('Missing or non-finite preflight gradient')
                if not bool(next(p for p in candidate.parameters() if p.ndim==4).grad.abs().sum()>0):
                    raise RuntimeError('First convolution receives no gradient')
                norm = torch.nn.utils.clip_grad_norm_(candidate.parameters(),5.,error_if_nonfinite=True)
                optimizer.step()
                assert all(torch.equal(v.detach().cpu(),frozen[n]) for n,v in candidate.named_buffers())
                torch.cuda.synchronize()
                rows.append(dict(loss=float(loss),grad_norm=float(norm),seconds=time.monotonic()-begin))
                del gradients,logits,loss
            records.append(dict(method=method,steps=rows,all_gradients_finite=True,
                                native_hard_forward_equal=True,source_buffers_fixed=True,
                                peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2))
            del candidate,optimizer,frozen
            gc.collect()
            torch.cuda.empty_cache()
        write(args.output/'preflight.json',dict(status='passed',physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],
                                               batch_size=args.batch_size,source_sha256=digest(args.source),arms=records))
    else:
        if args.source is None or args.method is None:
            parser.error('SNN requires a common source and method')
        model = make_ann(runtime, 2).to(args.device)
        state = torch.load(args.source, map_location='cpu', weights_only=False)
        if state.get('epoch') != 300 or state.get('seed') != SEED:
            raise ValueError('Expected the completed common source')
        model.load_state_dict(state['model'], strict=True)
        direct = dict(ann=evaluate(model,test,test_labels,args.eval_batch))
        native = runtime.model.resnet18(spike=True,bit=2,num_classes=100).to(args.device)
        native.load_state_dict(model.state_dict(),strict=True)
        direct['native_fast_snn'] = evaluate(native,test,test_labels,args.eval_batch)
        del native,state
        converted = FastClassifier(model,runtime,args.T,args.method).to(args.device)
        del model
        direct['adapter_hard'] = evaluate(converted,test,test_labels,args.eval_batch)
        write(args.output/'direct_conversion.json',direct)
        configure(args.device,SEED,4)
        fit(converted,raw,targets,test,test_labels,args,args.output/args.method,args.epochs,.01,5e-4,True)


if __name__ == '__main__':
    main()
