"""Pretrained Tiny200 ANN adaptation, QCFS source training, and direct ECL test."""
from __future__ import annotations
import argparse
import copy
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
from torch.utils.data import DataLoader, Subset
from src.tiny_imagenet import (SEED, TinyImages, TinyECL, make_model,
    pretrained_initialization, install_qcfs, calibrate_thresholds, sha256, AuthorQCFS)


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str)+'\n'); tmp.replace(path)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state())


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state(state['cuda'])


def load_source(path):
    raw = torch.load(path, map_location='cpu', weights_only=False)
    net = make_model(raw['architecture'])
    if raw['stage'] == 'qcfs': install_qcfs(net, raw['L'], raw['thresholds'])
    net.load_state_dict(raw['model'], strict=True)
    return net, raw


@torch.no_grad()
def evaluate(model, loader, device, output=None):
    training = model.training
    model.eval(); correct=top5=count=0; loss=0.; started=time.time()
    old_tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    try:
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            logits=model(images).float()
            if not bool(torch.isfinite(logits).all()): raise RuntimeError('Nonfinite validation logits')
            correct += int((logits.argmax(1)==labels).sum()); count += len(labels)
            top5 += int((logits.topk(5,dim=1).indices == labels[:,None]).any(1).sum())
            loss += float(nn.functional.cross_entropy(logits, labels, reduction='sum'))
        if count != 10000: raise RuntimeError('Incomplete Tiny validation')
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old_tf32
        model.train(training)
    result=dict(dataset='Tiny-ImageNet-200 official validation', total=count, correct=correct,
                top1=100*correct/count, top5=100*top5/count, ce=loss/count,
                seconds=time.time()-started, precision='FP32 TF32 disabled')
    if output: write(output,result)
    return result


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--architecture',choices=('resnet34','vgg16'),required=True)
    p.add_argument('--stage',choices=('relu','qcfs'),required=True)
    p.add_argument('--initializer',type=Path)
    p.add_argument('--source',type=Path)
    p.add_argument('--resume',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,required=True)
    p.add_argument('--batch-size',type=int,required=True)
    p.add_argument('--lr',type=float,required=True)
    p.add_argument('--L',type=int,default=4)
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--memory-cap-gib',type=float,required=True)
    p.add_argument('--eval-every',type=int,default=10)
    p.add_argument('--profile-only',action='store_true')
    p.add_argument('--max-updates',type=int)
    a=p.parse_args()
    physical=os.environ.get('CUDA_VISIBLE_DEVICES')
    if physical not in ('0','2','3'): p.error('Expose one authorized GPU: 0,2,or3')
    if a.output.exists() and any(a.output.iterdir()) and not a.resume:
        p.error('Fresh output or explicit resume required')
    a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
    device=torch.device('cuda:0')
    free,total=torch.cuda.mem_get_info(); cap=int(a.memory_cap_gib*1024**3)
    if cap > free-512*1024**2: raise RuntimeError('Insufficient free memory for the recorded allocator cap')
    torch.cuda.set_per_process_memory_fraction(cap/total)
    manifest=json.loads(a.manifest.read_text())
    if manifest['training_images']!=100000 or manifest['validation_images']!=10000:
        raise ValueError('Unexpected Tiny official split sizes')
    train=TinyImages(manifest,'train',training=True)
    validation=TinyImages(manifest,'validation')
    val_loader=DataLoader(validation,batch_size=128,num_workers=a.workers,pin_memory=True)
    thresholds={}; initial={}; progress=dict(completed_epochs=0,within_epoch_images=0,training_images=0,updates=0)
    if a.resume:
        model,old=load_source(a.resume)
        if (old['architecture'],old['stage'],old['manifest_sha256'],old['L']) != (a.architecture,a.stage,manifest['manifest_sha256'],a.L):
            raise ValueError('Resume provenance differs')
        thresholds=old['thresholds'];initial=old['initialization'];progress=old['progress']
    elif a.stage=='relu':
        if not a.initializer: p.error('ReLU source requires public initializer')
        model,initial=pretrained_initialization(a.architecture,a.initializer)
    else:
        if not a.source: p.error('QCFS adaptation requires this network\'s trained ReLU source')
        model,raw=load_source(a.source)
        if raw['stage']!='relu' or raw['architecture']!=a.architecture or raw['manifest_sha256']!=manifest['manifest_sha256']:
            raise ValueError('QCFS source provenance differs')
        initial=dict(parent_checkpoint=str(a.source.resolve()),parent_sha256=sha256(a.source),parent=raw['initialization'])
        model.to(device).eval()
        generator=torch.Generator().manual_seed(SEED)
        positions={label:[] for label in range(200)}
        for index,(_path,label) in enumerate(manifest['train']):positions[label].append(index)
        calibration_indices=[positions[label][int(j)] for label in range(200)
            for j in torch.randperm(len(positions[label]),generator=generator)[:5]]
        initial['calibration']=dict(images=1000,images_per_class=5,classes=200,
            selection='fixed-seed stratified training subset',seed=SEED,
            indices_sha256=hashlib.sha256(json.dumps(calibration_indices).encode()).hexdigest())
        calibration=DataLoader(Subset(TinyImages(manifest,'train'),calibration_indices),batch_size=32,num_workers=a.workers)
        thresholds=calibrate_thresholds(model,calibration,a.L)
        model.cpu();install_qcfs(model,a.L,thresholds)
    model=model.to(device)
    optimizer=torch.optim.SGD(model.parameters(),lr=a.lr,momentum=.9,weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda', enabled=False)
    if a.resume:
        optimizer.load_state_dict(old['optimizer']);scaler.load_state_dict(old['scaler']);restore_rng(old['rng'])
    runtime=dict(pid=os.getpid(),physical_gpu=physical,batch_size=a.batch_size,stage=a.stage,
                 architecture=a.architecture,started_at=time.time(),command=sys.argv)
    runtime['start_ticks']=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]
    write(a.output/'runtime.json',runtime)
    config=vars(a).copy();config.update(seed=SEED,physical_gpu=physical,manifest_sha256=manifest['manifest_sha256'],
        training_precision='AMP bfloat16, FP32 parameters; no loss scaling',eval_precision='FP32 TF32 disabled',
        per_sample_augmentation='seed+1000003*epoch+manifest_index; independent of batch/workers',
        adaptation='Tiny64 input,200 classes; transferable ImageNet initialization only',
        code_sha256={str(x.relative_to(ROOT)):sha256(x) for x in (Path(__file__).resolve(),ROOT/'src/tiny_imagenet.py')})
    if not a.resume:write(a.output/'config.json',config)
    else:
        with (a.output/'resume_events.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),checkpoint=str(a.resume),batch_size=a.batch_size,progress=progress))+'\n')
    stop=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    def status(state,**extra):
        write(a.output/'status.json',dict(status=state,state=state,pid=os.getpid(),physical_gpu=physical,
              batch_size=a.batch_size,stage=a.stage,architecture=a.architecture,epochs=a.epochs,
              planned_epochs=a.epochs,seed=SEED,L=a.L,updated_at=time.time(),
              epoch=progress['completed_epochs']+int(progress['within_epoch_images']>0),**progress,**extra))
    def save(name):
        raw=dict(format='tiny_imagenet_source_v1',architecture=a.architecture,stage=a.stage,L=a.L,
            manifest_sha256=manifest['manifest_sha256'],thresholds=thresholds,initialization=initial,
            model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
            progress=copy.deepcopy(progress),config=config,rng=rng_state())
        path=a.output/'checkpoints'/name;path.parent.mkdir(exist_ok=True)
        temp=path.with_suffix('.tmp');torch.save(raw,temp);temp.replace(path)
        return path
    started=time.time();invocation_updates=0;measured_images=0;measured_seconds=0.;last_checkpoint=None
    try:
        while progress['completed_epochs']<a.epochs:
            epoch=progress['completed_epochs']+1;train.epoch=epoch
            order=torch.randperm(len(train),generator=torch.Generator().manual_seed(SEED+epoch)).tolist()
            remaining=order[progress['within_epoch_images']:]
            loader=DataLoader(train,batch_size=a.batch_size,sampler=remaining,num_workers=a.workers,
                pin_memory=True,drop_last=False,**({'prefetch_factor':2} if a.workers else {}))
            model.train();epoch_started=time.time();loss_sum=0.;seen=correct=0
            warmup=min(1.,epoch/5);cosine=.5*(1+math.cos(math.pi*max(0,epoch-6)/max(1,a.epochs-5)))
            for group in optimizer.param_groups:group['lr']=a.lr*warmup*cosine
            for images,labels in loader:
                tick=time.time();images=images.to(device,non_blocking=True);labels=labels.to(device,non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits=model(images);loss=nn.functional.cross_entropy(logits,labels)
                if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite training loss')
                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                if invocation_updates==0:
                    missing=[n for n,p in model.named_parameters() if p.requires_grad and p.grad is None]
                    if missing:raise RuntimeError('Missing trainable gradients: '+str(missing))
                torch.cuda.synchronize()
                if a.profile_only:
                    r=dict(status='complete',architecture=a.architecture,stage=a.stage,batch_size=len(images),
                        physical_gpu=physical,seconds=time.time()-tick,loss=float(loss),gradient_norm=float(norm),
                        peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
                        peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,optimizer_updates=0)
                    write(a.output/'profile.json',r);print(json.dumps(r),flush=True);return
                scale_before=scaler.get_scale();scaler.step(optimizer);scaler.update()
                if scaler.get_scale()<scale_before:raise RuntimeError('AMP skipped an optimizer update')
                for node in model.modules():
                    if isinstance(node,AuthorQCFS):node.thresh.data.clamp_(min=1e-4)
                torch.cuda.synchronize();seconds=time.time()-tick;batch=len(images)
                progress['within_epoch_images']+=batch;progress['training_images']+=batch;progress['updates']+=1
                invocation_updates+=1;seen+=batch;loss_sum+=float(loss)*batch;correct+=int((logits.argmax(1)==labels).sum())
                if invocation_updates>5:measured_images+=batch;measured_seconds+=seconds
                if invocation_updates==1 or invocation_updates%10==0:
                    rate=measured_images/measured_seconds if measured_seconds else batch/max(seconds,1e-9)
                    status('running',last_loss=float(loss),images_per_second=rate,
                        estimated_train_epoch_seconds=100000/rate,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
                        peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20)
                if progress['within_epoch_images']==len(train):break
                if stop[0] or (a.max_updates and invocation_updates>=a.max_updates):break
            if progress['within_epoch_images']==len(train):
                progress['completed_epochs']+=1;progress['within_epoch_images']=0
                last_checkpoint=save('last.pt')
                if epoch%30==0 or epoch==a.epochs:save('epoch'+str(epoch)+'.pt')
                row=dict(epoch=epoch,train_loss=loss_sum/max(1,seen),train_accuracy=100*correct/max(1,seen),
                    epoch_seconds=time.time()-epoch_started,lr=optimizer.param_groups[0]['lr'],**progress)
                if epoch%a.eval_every==0 or epoch==a.epochs:
                    row['validation']=evaluate(model,val_loader,device,a.output/f'test_epoch{epoch}.json')
                with (a.output/'epochs.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
            if stop[0] or (a.max_updates and invocation_updates>=a.max_updates):break
        if not last_checkpoint or progress['within_epoch_images']:last_checkpoint=save('last.pt')
        complete=progress['completed_epochs']==a.epochs
        result=dict(status='complete' if complete else 'interrupted',architecture=a.architecture,stage=a.stage,
            epochs=a.epochs,**progress,batch_size=a.batch_size,physical_gpu=physical,
            checkpoint=str(last_checkpoint.resolve()),checkpoint_sha256=sha256(last_checkpoint),
            elapsed_seconds=time.time()-started,optimizer_updates_counted=True)
        if complete and a.stage=='qcfs':
            result['source_ANN']=json.loads((a.output/f'test_epoch{a.epochs}.json').read_text())
            write(a.output/'ann_validation.json',result['source_ANN'])
            result['ECL']={}
            model=model.cpu().eval();torch.cuda.empty_cache()
            for T in (2,4):
                snn=TinyECL(model,T).to(device)
                result['ECL'][str(T)]=evaluate(snn,val_loader,device,a.output/f'ecl_T{T}.json')
                del snn;torch.cuda.empty_cache()
        write(a.output/'summary.json',result);status(result['status'],checkpoint=str(last_checkpoint))
        print(json.dumps(result),flush=True)
    except BaseException as e:
        status('error',error=repr(e));raise


if __name__=='__main__':main()
