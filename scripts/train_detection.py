"""Paired Fast-SNN VOC experiments with one hard signed-ECL trajectory."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import copy
import json
import math
import os
import random
import subprocess
import time

import numpy as np
import torch

from src.detection_fast_snn import upstream_runtime, build_detector, make_evaluator
from src.runtime import configure, cpu_state, digest, generators, report, restore_rng, rng_state, save


def parser():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('mode', choices=('train', 'evaluate'))
    p.add_argument('--upstream', required=True, help='Fast-SNN/object detection directory')
    p.add_argument('--source', required=True, help='Original author VOC QAT checkpoint')
    p.add_argument('--data', required=True, help='VOCdevkit directory')
    p.add_argument('--output', required=True)
    p.add_argument('--method', choices=('ann','author','hard','triangular','phase_triangular_density'),
                   default='phase_triangular_density')
    p.add_argument('--checkpoint', help='Adapted SNN checkpoint for evaluation')
    p.add_argument('--L', type=int, default=3)
    p.add_argument('--T', type=int, default=3)
    p.add_argument('--size', type=int, default=416)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--min-lr', type=float, default=.00001)
    p.add_argument('--warmup-epochs', type=int, default=2)
    p.add_argument('--official-test-epochs', type=int, nargs='*', default=[10,20])
    p.add_argument('--seed', type=int, default=2026100321)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--resume', action='store_true')
    return p


def seed_worker(_):
    import cv2
    cv2.setNumThreads(0)
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


def test(runtime, model, data, output, device, size, checkpoint=None):
    model.trainable = False
    model.set_grid(size)
    model.eval()
    evaluator = make_evaluator(runtime, data, output, device, size)
    start = time.monotonic()
    with torch.no_grad():
        evaluator.evaluate(model)
    result = dict(metrics=dict(map50=float(evaluator.map)*100,
                               count=len(evaluator.dataset), protocol='VOC2007_11point_AP50'),
                  seconds=time.monotonic()-start)
    if checkpoint:
        result.update(checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint))
    report(result, Path(output)/'test.json')
    return result


def learning_rate(cfg, epoch):
    if epoch <= cfg['warmup_epochs']:
        return cfg['lr']*epoch/cfg['warmup_epochs']
    fraction=(epoch-cfg['warmup_epochs'])/(cfg['epochs']-cfg['warmup_epochs'])
    return cfg['min_lr']+.5*(cfg['lr']-cfg['min_lr'])*(1+math.cos(math.pi*fraction))


def main():
    args=parser().parse_args()
    exponent=math.log2(args.L+1)
    if not exponent.is_integer() or args.L < 1 or args.T < 1:
        raise ValueError('Fast-SNN sources require L=2**b-1 and positive T')
    bit=int(exponent)
    if args.mode=='train' and args.method not in ('triangular','phase_triangular_density'):
        raise ValueError('Train a matched triangular or shared-phase arm')
    device=configure(args.device,args.seed,args.threads)
    output=Path(args.output).resolve()
    runtime=upstream_runtime(args.upstream)
    cfg={k:v for k,v in vars(args).items() if k not in ('mode','checkpoint','resume','device','output')}
    for key in ('source','data','upstream'):
        cfg[key]=str(Path(cfg[key]).resolve())
    cfg.update(source_sha256=digest(args.source), source_bit=bit,
               architecture='Fast-SNN_YOLOv2_ResNet34', momentum=.9,weight_decay=.0005,
               gradient_clip=5.,event_rule='standard signed binary ECL; positive > threshold',
               upstream_commit=subprocess.check_output(
                   ['git','-C',str(runtime.path),'rev-parse','HEAD'],text=True).strip())
    source=build_detector(runtime,args.source,bit=bit,timesteps=args.T,method=args.method,
                          size=args.size,device=device,trainable=args.mode=='train')
    if args.mode=='evaluate':
        if args.checkpoint:
            saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
            if saved['config']['source_sha256']!=cfg['source_sha256'] or saved['config']['T']!=args.T:
                raise ValueError('Evaluation checkpoint source or T mismatch')
            source.load_state_dict(saved['model'],strict=True)
        output.mkdir(parents=True,exist_ok=True)
        save(output/'config.json',cfg,json_format=True)
        test(runtime,source,args.data,output,device,args.size,args.checkpoint or args.source)
        return
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError('A new training arm requires an empty output directory')
    output.mkdir(parents=True,exist_ok=True)
    if args.resume:
        previous=json.loads((output/'config.json').read_text())
        if previous!=cfg:
            raise ValueError('Resume must retain the original training configuration')
    else:
        save(output/'config.json',cfg,json_format=True)
    data=runtime.data.VOCDetection(data_dir=str(Path(args.data).resolve()),
                  transform=runtime.augment.SSDAugmentation(args.size))
    if len(data)!=16551:
        raise ValueError(f'Expected all 16551 VOC07+12 trainval images, got {len(data)}')
    streams=generators(args.seed)
    loader=torch.utils.data.DataLoader(data,batch_size=args.batch_size,shuffle=True,
            num_workers=args.workers,collate_fn=runtime.data.detection_collate,
            generator=streams['order'],worker_init_fn=seed_worker,pin_memory=device.type=='cuda',
            drop_last=False)
    optimizer=torch.optim.SGD(source.parameters(),lr=args.lr,momentum=.9,weight_decay=.0005)
    epoch0,steps,total_seconds=0,0,0.
    if args.resume:
        saved=torch.load(output/'latest.pt',map_location='cpu',weights_only=False)
        source.load_state_dict(saved['model'],strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        epoch0,steps,total_seconds=saved['epoch'],saved['steps'],saved['training_seconds']
        restore_rng(saved['rng'],streams)
    initial_thresholds={n:v.detach().cpu().clone() for n,v in source.named_buffers()
                        if n.endswith('.threshold')}
    if len(initial_thresholds)!=38:
        raise ValueError('Expected 38 fixed spiking thresholds')
    save(output/'status.json',dict(status='running',pid=os.getpid(),epoch=epoch0,
         physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),batch_size=args.batch_size),json_format=True)
    row=(json.loads((output/'epochs.jsonl').read_text().splitlines()[-1]) if epoch0 else None)
    for epoch in range(epoch0+1,args.epochs+1):
        source.train()
        source.trainable=True
        source.set_grid(args.size)
        lr=learning_rate(cfg,epoch)
        for group in optimizer.param_groups:
            group['lr']=lr
        start=time.monotonic()
        losses=np.zeros(5,dtype=np.float64)
        examples=0
        for batch_index,(images,labels) in enumerate(loader,1):
            labels=runtime.tools.gt_creator(input_size=args.size,stride=source.stride,
                    label_lists=[label.tolist() for label in labels],anchor_size=runtime.config.yolov2_r50_cfg['anchor_size_voc'])
            images=images.to(device,non_blocking=True)
            labels=torch.as_tensor(labels,dtype=torch.float32,device=device)
            optimizer.zero_grad(set_to_none=True)
            components=source(images,target=labels)
            total=sum(components)
            if not bool(torch.isfinite(total)):
                raise RuntimeError('Non-finite detection loss; no image batch was skipped')
            total.backward()
            norm=torch.nn.utils.clip_grad_norm_(source.parameters(),5.,error_if_nonfinite=True)
            optimizer.step()
            values=torch.stack((*components,total)).detach().double().cpu().numpy()
            losses+=values*len(images)
            examples+=len(images)
            steps+=1
            if batch_index%50==0 or batch_index==len(loader):
                progress=dict(event='training_progress',epoch=epoch,batch=batch_index,batches=len(loader),
                              steps=steps,training_examples=examples,epoch_seconds_elapsed=time.monotonic()-start,
                              mean_total_loss=float(losses[-1]/examples))
                print(json.dumps(progress,allow_nan=False),flush=True)
                save(output/'status.json',dict(status='running',pid=os.getpid(),
                     physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),**progress),json_format=True)
        if device.type=='cuda':
            torch.cuda.synchronize(device)
        elapsed=time.monotonic()-start
        total_seconds+=elapsed
        assert examples==16551
        row=dict(epoch=epoch,steps=steps,training_examples=examples,lr=lr,
                 epoch_seconds=elapsed,training_seconds=total_seconds,batch_size=args.batch_size,
                 gradient_norm=float(norm),**dict(zip(('conf','cls','box','iou','total_loss'),losses/examples)))
        with (output/'epochs.jsonl').open('a') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row,allow_nan=False),flush=True)
        for name,value in source.named_buffers():
            if name in initial_thresholds:
                assert torch.equal(value.detach().cpu(),initial_thresholds[name])
        payload=dict(config=cfg,model=cpu_state(source),optimizer=copy.deepcopy(optimizer.state_dict()),
                     epoch=epoch,steps=steps,training_seconds=total_seconds,rng=rng_state(streams))
        save(output/'latest.pt',payload)
        if epoch%5==0 or epoch==args.epochs or epoch in args.official_test_epochs:
            save(output/f'epoch{epoch}.pt',payload)
        save(output/'status.json',dict(status='running',pid=os.getpid(),epoch=epoch,steps=steps,
             physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),batch_size=args.batch_size),json_format=True)
        if epoch==args.epochs or epoch in args.official_test_epochs:
            result=test(runtime,source,args.data,output/f'official_epoch{epoch}',device,args.size,output/f'epoch{epoch}.pt')
            save(output/f'test_epoch{epoch}.json',result,json_format=True)
            if epoch==args.epochs:
                save(output/'test.json',result,json_format=True)
        del payload
    if not (output/'test.json').exists():
        result=test(runtime,source,args.data,output/f'official_epoch{args.epochs}',device,args.size,output/f'epoch{args.epochs}.pt')
        save(output/f'test_epoch{args.epochs}.json',result,json_format=True)
        save(output/'test.json',result,json_format=True)
    save(output/'summary.json',dict(status='complete',final=row,config=cfg,
         image_presentations=16551*args.epochs,peak_allocated_mib=(torch.cuda.max_memory_allocated(device)/2**20
         if device.type=='cuda' else None)),json_format=True)
    save(output/'status.json',dict(status='complete',pid=os.getpid(),epoch=args.epochs,steps=steps,
         official_test=json.loads((output/'test.json').read_text())['metrics']),json_format=True)


if __name__=='__main__':
    main()
