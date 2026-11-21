#!/usr/bin/env python3
"""Real DiMP18 pilot: task-adapt one QCFS source, then matched SNN arms.

Uses the official DiMPProcessing, full five-step native DiMPActor and native
PrROI. This pilot checks the pipeline; it does not qualify a source model.
"""
from __future__ import annotations
import argparse
import contextlib
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
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PREP = ROOT/'artifacts/dimp18_preparation_20261119'
sys.path.insert(0, str(PREP/'vendor'))
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from src.tracking_dimp import DimpFeatureBackbone
from src.training_dimp import (DEFAULT_UPSTREAM, DEFAULT_OFFICIAL, initialized_model,
                              model_from_checkpoint, converted_model_from_source,
                              warm_started_model, freeze_batchnorm, file_sha256)
from src.conversion.qcfs_ecl_resnet18 import AuthorQCFS


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str)+'\n')
    temporary.replace(path)


def json_append(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, default=str)+'\n')


def state_digest(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode()); h.update(str(value.dtype).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@contextlib.contextmanager
def item_rng(seed):
    states = random.getstate(), np.random.get_state(), torch.get_rng_state()
    random.seed(seed); np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(states[0]); np.random.set_state(states[1]); torch.set_rng_state(states[2])


def make_processing(image_size=288):
    from ltr.data import processing
    import ltr.data.transforms as tfm
    return processing.DiMPProcessing(
        search_area_factor=5., output_sz=image_size,
        center_jitter_factor={'train':3., 'test':4.5},
        scale_jitter_factor={'train':.25, 'test':.5}, mode='sequence',
        proposal_params={'min_iou':.1, 'boxes_per_frame':8,
                         'sigma_factor':[.01,.05,.1,.2,.3]},
        label_function_params={'feature_sz':image_size//16,
                               'sigma_factor':.25/5., 'kernel_sz':4},
        transform=tfm.Transform(tfm.ToTensorAndJitter(.2),
                                 tfm.Normalize(mean=[.485,.456,.406], std=[.229,.224,.225])),
        joint_transform=tfm.Transform(tfm.ToGrayscale(probability=.05)))


class FixedPairDataset(Dataset):
    def __init__(self, pairs, seed, epoch, image_size):
        self.pairs, self.seed, self.epoch = pairs, seed, epoch
        self.processing = make_processing(image_size)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        import cv2
        from pytracking import TensorDict
        cv2.setNumThreads(0)
        pair = self.pairs[index]
        seed = (self.seed + self.epoch*1_000_003 + index*10_007) % (2**32-1)
        with item_rng(seed):
            images, boxes = [], []
            for branch in ('template','search'):
                value = cv2.imread(pair[branch]['image'])
                if value is None:
                    raise ValueError('Cannot read '+pair[branch]['image'])
                images.append(cv2.cvtColor(value, cv2.COLOR_BGR2RGB))
                boxes.append(torch.tensor(pair[branch]['bbox_xywh'], dtype=torch.float32))
            result = self.processing(TensorDict(train_images=[images[0]], test_images=[images[1]],
                                                 train_anno=[boxes[0]], test_anno=[boxes[1]]))
        if not all(isinstance(v, torch.Tensor) and bool(torch.isfinite(v).all()) for v in result.values()):
            raise RuntimeError('Non-finite or non-tensor official processing output')
        digest = state_digest(result)
        return dict(data=dict(result), pair_index=index, augmentation_seed=seed,
                    pair_id=pair['pair_id'], augmentation_digest=digest)


def collate_pairs(samples):
    data = {key:torch.stack([sample['data'][key] for sample in samples], dim=1)
            for key in samples[0]['data']}
    data['_pair_indices'] = [sample['pair_index'] for sample in samples]
    data['_pair_ids'] = [sample['pair_id'] for sample in samples]
    data['_augmentation_seeds'] = [sample['augmentation_seed'] for sample in samples]
    data['_augmentation_digests'] = [sample['augmentation_digest'] for sample in samples]
    return data


def optimizer_for(net):
    groups = [
        ('filter_initializer', net.classifier.filter_initializer.parameters(), 5e-5),
        ('filter_optimizer', net.classifier.filter_optimizer.parameters(), 5e-4),
        ('classification_features', net.classifier.feature_extractor.parameters(), 5e-5),
        ('iou_head', net.bb_regressor.parameters(), 1e-3),
        ('backbone', net.feature_extractor.parameters(), 2e-4)]
    return torch.optim.Adam([dict(params=[p for p in params if p.requires_grad], lr=lr, name=name)
                             for name, params, lr in groups])


def group_hashes(net):
    groups = {'backbone':net.feature_extractor.state_dict(),
              'classifier':net.classifier.state_dict(), 'iou_head':net.bb_regressor.state_dict()}
    return {name:state_digest(value) for name,value in groups.items()}


def snapshot_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(value):
    random.setstate(value['python']); np.random.set_state(value['numpy'])
    torch.set_rng_state(value['torch']); torch.cuda.set_rng_state_all(value['cuda'])


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--method', choices=['source','triangular','density'], required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--pairs', type=Path, default=PREP/'data/train_pairs.jsonl')
    ap.add_argument('--initialization', type=Path, default=PREP/'conversion/qcfs_initialization.pt')
    ap.add_argument('--source-checkpoint', type=Path)
    ap.add_argument('--upstream', type=Path, default=DEFAULT_UPSTREAM)
    ap.add_argument('--official-checkpoint', type=Path, default=DEFAULT_OFFICIAL)
    ap.add_argument('--resume', type=Path)
    ap.add_argument('--warm-start', type=Path, help='Inherit all weights, reset optimizer/progress/RNG')
    ap.add_argument('--purpose', choices=['pilot_pipeline_only','full_tracking'], default='pilot_pipeline_only')
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--max-pairs', type=int, default=1024)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--image-size', type=int, default=288)
    ap.add_argument('--seed', type=int, default=2026100321)
    ap.add_argument('--T', type=int, default=4)
    ap.add_argument('--memory-cap-gib', type=float)
    ap.add_argument('--max-steps', type=int, help='Stop and checkpoint without replaying consumed pairs on resume')
    ap.add_argument('--checkpoint-every', type=int, default=0)
    ap.add_argument('--save-every-epochs', type=int, default=1, help='Keep evaluation anchors; last.pt is saved every epoch')
    ap.add_argument('--profile-only', action='store_true', help='One actual F+B, no optimizer step or training images counted')
    args = ap.parse_args()
    if args.resume and args.warm_start:
        ap.error('--resume and --warm-start have different semantics; choose one')
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('0','2','3'):
        ap.error('Expose exactly one authorized physical GPU: 0, 2, or 3')
    if args.image_size != 288 or args.T != 4 or args.seed != 2026100321:
        ap.error('This fixed pilot uses crop288, T4 and seed2026100321')
    if args.epochs < 1 or args.max_pairs < 1 or args.batch_size < 1:
        ap.error('Positive epochs, pair count and batch required')
    if args.save_every_epochs < 1:
        ap.error('--save-every-epochs must be positive')
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        ap.error('Use a fresh output or explicit --resume to preserve prior evidence')
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoints = args.output/'checkpoints'; checkpoints.mkdir(exist_ok=True)
    sys.path.insert(0, str(args.upstream.resolve()))
    from ltr.actors.tracking import DiMPActor
    from ltr.models.loss.target_classification import LBHinge
    torch.set_num_threads(2)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(False) # Native PrROI backward uses atomics.
    assert torch.cuda.device_count()==1
    device=torch.device('cuda:0'); free,total=torch.cuda.mem_get_info(0)
    cap_bytes = min(int(args.memory_cap_gib*1024**3) if args.memory_cap_gib else free-2*1024**3,
                    free-2*1024**3)
    if cap_bytes < 2*1024**3:
        raise RuntimeError('Not enough free memory after the fixed 2GiB reserve')
    torch.cuda.set_per_process_memory_fraction(cap_bytes/total,0)
    pairs=[]
    with args.pairs.open() as stream:
        for line in stream:
            if len(pairs)==args.max_pairs:
                if args.purpose=='full_tracking' and line.strip():
                    raise ValueError('Full tracking must consume the complete frozen pair manifest')
                break
            pairs.append(json.loads(line))
    if len(pairs)!=args.max_pairs:
        raise ValueError('Fixed manifest contains too few pairs')
    selection_sha=hashlib.sha256('\n'.join(p['pair_id'] for p in pairs).encode()).hexdigest()
    config=vars(args).copy(); config.update(physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],logical_gpu=0,
        memory_before_free_MiB=free/1024**2,memory_cap_MiB=cap_bytes/1024**2,
        source_quality_qualified=False,purpose=args.purpose,support_frames=1,search_frames=1,
        pairs_selection_sha256=selection_sha,pairs_manifest_sha256=file_sha256(args.pairs),
        native_processing='DiMPProcessing author jitter/ToGrayscale/ToTensorAndJitter/8 proposals/Gaussian',
        proposal_iou_target='2*actual_IoU-1',global_BN_statistics='frozen',
        native_loss='LBHinge localization classifier + IoU MSE (not identity CE)',
        planned_pairs_per_epoch=args.max_pairs,planned_images_per_epoch=2*args.max_pairs,
        planned_optimizer_steps_per_epoch=math.ceil(args.max_pairs/args.batch_size))
    config['code_sha256']={str(path.relative_to(ROOT)):file_sha256(path) for path in
                          [Path(__file__).resolve(), ROOT/'src/training_dimp.py',ROOT/'src/tracking_dimp.py']}
    if args.resume:
        old=torch.load(args.resume,map_location='cpu',weights_only=False)
        for field in ['method','max_pairs','seed','T','image_size','pairs_selection_sha256','pairs_manifest_sha256','purpose']:
            if str(old['config'][field]) != str(config[field]):
                raise ValueError('Resume configuration differs: '+field)
        net,metadata=model_from_checkpoint(args.resume)
        progress=copy.deepcopy(old['progress']); accumulator=copy.deepcopy(old['epoch_accumulator'])
        initial_groups=old['initial_parameter_group_hashes']
    else:
        if args.warm_start:
            net,metadata=warm_started_model(args.warm_start)
        elif args.method=='source':
            net,metadata=initialized_model(args.initialization,args.upstream,args.official_checkpoint)
        else:
            if not args.source_checkpoint and args.profile_only:
                net,metadata=initialized_model(args.initialization,args.upstream,args.official_checkpoint)
                net.feature_extractor=DimpFeatureBackbone(net.feature_extractor.source,args.T,args.method)
                metadata.update(method=args.method,stage='untrained_initialization_profile',source_is_trained=False)
            elif not args.source_checkpoint:
                ap.error('SNN arms require --source-checkpoint')
            else:
                net,metadata=converted_model_from_source(args.source_checkpoint,args.method,args.T)
        progress=dict(completed_epochs=0,within_epoch_pairs=0,total_pairs=0,actual_images=0,optimizer_steps=0)
        accumulator=dict(weighted_stats={},pairs=0,augmentation_records=[],epoch_pair_indices=[])
        initial_groups=group_hashes(net)
    if metadata['method']!=args.method:
        raise ValueError('Method mismatch')
    prefix='full' if args.purpose=='full_tracking' else 'pilot'
    metadata.update(format='dimp18_training_v1' if args.purpose=='full_tracking' else 'dimp18_pilot_v1',
                    purpose=args.purpose,scope='complete_fixed_training_manifest' if args.purpose=='full_tracking' else 'fixed_first_1024_pairs_pilot',
                    planned_training_pairs_per_epoch=args.max_pairs,planned_training_epochs=args.epochs,
                    code_sha256=config['code_sha256'])
    config['warmup_extra_training_images']=metadata.get('warmup_extra_training_images',0)
    net=net.to(device).train(); freeze_batchnorm(net)
    metadata['T']=args.T
    optimizer=optimizer_for(net)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=15,gamma=.2)
    if args.resume:
        optimizer.load_state_dict(old['optimizer']); scheduler.load_state_dict(old['scheduler']); restore_rng(old['rng'])
    actor=DiMPActor(net,{'iou':nn.MSELoss(),'test_clf':LBHinge(threshold=.05)},
                    {'iou':1.,'test_clf':100.,'test_init_clf':100.,'test_iter_clf':400.})
    active=[(n,p) for n,p in net.named_parameters() if p.requires_grad]
    frozen_buffers={n:v.detach().clone() for n,v in net.named_buffers()}
    stop=[False]
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
    started=time.time(); completed_this_invocation=0
    runtime=dict(pid=os.getpid(),runtime_pid=os.getpid(),physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],
                 method=args.method,output=str(args.output.resolve()),started_at=started,command=sys.argv,
                 purpose=args.purpose,profile_only=args.profile_only)
    json_write(args.output/'runtime_pid.json',runtime)
    json_write(args.output/'runtime.json',runtime)
    json_append(args.output/'runtime_history.jsonl',runtime)
    if not args.resume:
        json_write(args.output/'config.json',config)
        json_write(args.output/'selected_pairs.json',[p['pair_id'] for p in pairs])
    else:
        json_append(args.output/'resume_events.jsonl',dict(time=time.time(),checkpoint=str(args.resume),progress=progress,pid=os.getpid(),previous_batch_size=old['config']['batch_size'],batch_size=args.batch_size))
    def status(state,**extra):
        value=dict(state=state,status=state,pid=os.getpid(),runtime_pid=os.getpid(),method=args.method,
                   stage=prefix+'_source_adaptation' if args.method=='source' else prefix+'_snn_finetuning',
                   purpose=args.purpose,warmup_extra_training_images=config['warmup_extra_training_images'],
                   source_quality_qualified=False,updated_at=time.time(),planned_epochs=args.epochs,
                   planned_pairs_per_epoch=args.max_pairs,planned_images_per_epoch=2*args.max_pairs,
                   **progress,**extra)
        json_write(args.output/'status.json',value)
    def save_checkpoint(filename):
        meta=copy.deepcopy(metadata)
        meta.update(stage=prefix+'_source_adapted' if args.method=='source' and progress['completed_epochs']>=args.epochs else
                    (prefix+'_source_adapting' if args.method=='source' else prefix+'_snn_finetuned'),
                    source_is_trained=bool(progress['optimizer_steps']>0 or metadata.get('source_is_trained')),
                    source_quality_qualified=False,model_state_sha256=state_digest(net.state_dict()))
        raw=dict(model=net.state_dict(),metadata=meta,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),config=config,
                 progress=copy.deepcopy(progress),epoch_accumulator=copy.deepcopy(accumulator),rng=snapshot_rng(),
                 initial_parameter_group_hashes=initial_groups,current_parameter_group_hashes=group_hashes(net),
                 saved_at=time.time())
        path=checkpoints/filename;temporary=path.with_name(path.name+'.tmp')
        torch.save(raw,temporary);temporary.replace(path)
        return dict(path=str(path.resolve()),sha256=file_sha256(path),model_state_sha256=meta['model_state_sha256'])
    try:
        status('running',initial_model_state_sha256=state_digest(net.state_dict()))
        first_gradient_check=None
        while progress['completed_epochs']<args.epochs:
            epoch=progress['completed_epochs']+1
            order=torch.randperm(len(pairs),generator=torch.Generator().manual_seed(args.seed+epoch)).tolist()
            remaining=order[progress['within_epoch_pairs']:]
            dataset=FixedPairDataset(pairs,args.seed,epoch,args.image_size)
            loader=DataLoader(dataset,batch_size=args.batch_size,sampler=remaining,num_workers=args.workers,
                              collate_fn=collate_pairs,drop_last=False,pin_memory=True,
                              persistent_workers=False,**({'prefetch_factor':2} if args.workers else {}))
            for batch in loader:
                tick=time.time(); count=len(batch['_pair_indices'])
                data={k:v.to(device,non_blocking=True) for k,v in batch.items() if not k.startswith('_')}
                optimizer.zero_grad(set_to_none=True)
                loss,stats=actor(data)
                if not bool(torch.isfinite(loss)): raise RuntimeError('Non-finite native tracking loss')
                loss.backward()
                absent=[n for n,p in active if p.grad is None]
                invalid=[n for n,p in active if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
                if absent or invalid: raise RuntimeError('Invalid gradients: '+str(dict(absent=absent,nonfinite=invalid)))
                norms={name:sum(float(p.grad.detach().square().sum()) for p in group['params'])**.5
                       for group in optimizer.param_groups for name in [group['name']]}
                if first_gradient_check is None:
                    first_gradient_check=dict(active_parameter_tensors=len(active),all_present_and_finite=True,
                        first_conv_grad_l1=float(net.feature_extractor.source.conv1.weight.grad.abs().sum()),group_gradient_norms=norms)
                    if not first_gradient_check['first_conv_grad_l1']>0: raise RuntimeError('Zero first-conv gradient')
                    json_write(args.output/'first_gradient_check.json',first_gradient_check)
                torch.cuda.synchronize()
                peak=dict(peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,
                          peak_reserved_MiB=torch.cuda.max_memory_reserved()/1024**2)
                if args.profile_only:
                    profile=dict(status='complete',method=args.method,batch_size=count,crop=288,T=args.T,
                        forward_backward_seconds=time.time()-tick,loss=float(loss),native_stats=stats,
                        gradient_check=first_gradient_check,physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],
                        memory_cap_MiB=cap_bytes/1024**2,**peak,training_images_counted=0,
                        preflight_image_visits=2*count,optimizer_step=False,source_quality_qualified=False)
                    json_write(args.output/'profile.json',profile);status('profile_complete',**peak)
                    print(json.dumps(profile),flush=True);return
                optimizer.step()
                if args.method=='source':
                    with torch.no_grad():
                        for module in net.feature_extractor.source.modules():
                            if isinstance(module,AuthorQCFS): module.thresh.clamp_(min=1e-4)
                changed_buffers=[n for n,v in net.named_buffers() if not torch.equal(v,frozen_buffers[n])]
                if changed_buffers: raise RuntimeError('Frozen BN/threshold buffers changed: '+str(changed_buffers))
                progress['within_epoch_pairs']+=count;progress['total_pairs']+=count
                progress['actual_images']+=2*count;progress['optimizer_steps']+=1;completed_this_invocation+=1
                accumulator['pairs']+=count
                for k,v in stats.items():accumulator['weighted_stats'][k]=accumulator['weighted_stats'].get(k,0.)+float(v)*count
                accumulator['augmentation_records'].extend(batch['_augmentation_digests'])
                accumulator['epoch_pair_indices'].extend(batch['_pair_indices'])
                record=dict(epoch=epoch,step=progress['optimizer_steps'],pairs=count,actual_images=2*count,
                    native_stats=stats,group_gradient_norms=norms,all_active_gradients_present_and_finite=True,
                    first_conv_grad_l1=float(net.feature_extractor.source.conv1.weight.grad.abs().sum()),
                    elapsed_seconds=time.time()-tick,**peak,pair_indices=batch['_pair_indices'],
                    augmentation_digest=hashlib.sha256(''.join(batch['_augmentation_digests']).encode()).hexdigest())
                json_append(args.output/'steps.jsonl',record)
                status('running',epoch=epoch,last_loss=float(loss),last_classification_LBHinge=stats['ClfTrain/test_loss'],
                       last_iou_loss=stats['Loss/iou'],**peak)
                if progress['within_epoch_pairs']==len(pairs):
                    epoch_record=dict(epoch=epoch,pairs=accumulator['pairs'],actual_images=2*accumulator['pairs'],
                        native_stats={k:v/accumulator['pairs'] for k,v in accumulator['weighted_stats'].items()},
                        pair_order_sha256=hashlib.sha256(json.dumps(accumulator['epoch_pair_indices']).encode()).hexdigest(),
                        augmentation_sha256=hashlib.sha256(''.join(accumulator['augmentation_records']).encode()).hexdigest(),
                        optimizer_steps=progress['optimizer_steps'],parameter_group_hashes=group_hashes(net),
                        frozen_buffers_unchanged=True,source_quality_qualified=False,**peak)
                    progress['completed_epochs']+=1;progress['within_epoch_pairs']=0
                    scheduler.step()
                    accumulator=dict(weighted_stats={},pairs=0,augmentation_records=[],epoch_pair_indices=[])
                    ck=save_checkpoint('last.pt')
                    if progress['completed_epochs']%args.save_every_epochs==0 or progress['completed_epochs']>=args.epochs:
                        ck=save_checkpoint('epoch'+str(progress['completed_epochs'])+'.pt')
                    epoch_record['checkpoint']=ck;json_append(args.output/'epochs.jsonl',epoch_record)
                    print(json.dumps(epoch_record),flush=True)
                    break
                if args.checkpoint_every and progress['optimizer_steps']%args.checkpoint_every==0:
                    save_checkpoint('last.pt')
                if stop[0] or (args.max_steps and completed_this_invocation>=args.max_steps):break
            if stop[0] or (args.max_steps and completed_this_invocation>=args.max_steps):break
        ck=save_checkpoint('last.pt')
        complete=progress['completed_epochs']>=args.epochs
        final_groups=group_hashes(net)
        summary=dict(status='complete' if complete else 'interrupted',purpose=args.purpose,method=args.method,
            stage=(prefix+'_source_adapted' if complete else prefix+'_source_adapting') if args.method=='source' else prefix+'_snn_finetuned',
            warmup_extra_training_images=config['warmup_extra_training_images'],
            source_quality_qualified=False,planned_epochs=args.epochs,planned_pairs_per_epoch=args.max_pairs,
            planned_total_images=2*args.max_pairs*args.epochs,**progress,checkpoint=ck,
            parameter_groups_updated={k:initial_groups[k]!=v for k,v in final_groups.items()},
            initial_parameter_group_hashes=initial_groups,final_parameter_group_hashes=final_groups,
            frozen_buffers_unchanged=True,elapsed_this_invocation_seconds=time.time()-started,
            peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,
            peak_reserved_MiB=torch.cuda.max_memory_reserved()/1024**2,
            physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],batch_size=args.batch_size,L=metadata['L'],T=args.T,
            first_gradient_check=first_gradient_check,resume_supported=True,
            next_action='awaiting_source_review' if args.method=='source' and complete and args.purpose=='full_tracking' else
                        ('matched pilot SNN arms' if args.method=='source' and complete else 'root evaluates complete SOT tracks'))
        json_write(args.output/'summary.json',summary);status(summary['status'],checkpoint=ck)
        print(json.dumps(summary),flush=True)
    except BaseException as error:
        status('error',error=repr(error),traceback=traceback.format_exc())
        print(traceback.format_exc(),flush=True)
        raise


if __name__=='__main__':main()
