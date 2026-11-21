"""Prepare original IBM Gesture labels and official trial split, without GPU."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
import inspect
import re
import tarfile
from urllib.parse import urlencode

import numpy as np
from scripts.prepare_dvs import digest, download_verified, write_json


PUBLIC_FILES = {'DvsGesture.tar.gz':'f_211521748942','README.txt':'f_211519208560',
                'LICENSE.txt':'f_199380302705','gesture_mapping.csv':'f_199380292206'}
SHARED = '3hiq58ww1pbbjrinh367ykfdf60xsfm8'


def load_gesture_names(path):
    names = {}
    with Path(path).open() as stream:
        for row in csv.DictReader(stream):
            target = int(row['label'])-1
            action = row['action'].strip()
            if not action:
                raise ValueError('Empty original gesture action')
            names.setdefault(target,[]).append(action)
    if set(names)!=set(range(11)):
        raise ValueError('Expected original11 gesture labels')
    # The original mapping assigns forward/backward forearm rolls to label8.
    # Retain both action names without inventing an extra classification label.
    return {target:' / '.join(actions) for target,actions in names.items()}


def prepare(root,T,workers):
    from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
    from spikingjelly.datasets import integrate_events_by_fixed_frames_number
    root = root.resolve()
    root.mkdir(parents=True,exist_ok=True)
    lock = (root/'.prepare.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    downloads,extract,frames_root = root/'download',root/'extract',root/f'frames_number_{T}'
    for folder in [downloads,extract,frames_root]:
        folder.mkdir(exist_ok=True)
    resources = []
    for name,page,md5 in DVS128Gesture.resource_url_md5():
        url = 'https://ibm.ent.box.com/index.php?'+urlencode(dict(
            rm='box_download_shared_file',shared_name=SHARED,file_id=PUBLIC_FILES[name]))
        download_verified(url,downloads/name,md5)
        resources.append(dict(name=name,official_page=page,url=url,md5=md5))
    data = extract/'DvsGesture'
    if not (extract/'complete.json').exists():
        with tarfile.open(downloads/'DvsGesture.tar.gz','r:gz') as archive:
            for member in archive.getmembers():
                target = (extract/member.name).resolve()
                if not target.is_relative_to(extract) or member.issym() or member.islnk():
                    raise ValueError('Unexpected archive path or link')
            archive.extractall(extract)
        if not (data/'trials_to_train.txt').is_file() or not (data/'trials_to_test.txt').is_file():
            raise ValueError('Original archive lacks official split lists')
        write_json(extract/'complete.json',dict(archive_md5=resources[0]['md5']))
    names = load_gesture_names(downloads/'gesture_mapping.csv')
    splits = {split:[x.strip() for x in (data/f'trials_to_{split}.txt').read_text().splitlines() if x.strip()]
              for split in ['train','test']}
    if set(splits['train']) & set(splits['test']):
        raise ValueError('Official trial lists overlap')
    subjects = {split:sorted({int(re.search(r'user(\d+)',x).group(1)) for x in files})
                for split,files in splits.items()}
    if set(subjects['train']) & set(subjects['test']):
        raise ValueError('Official subject split overlaps')
    parser_path = Path(inspect.getsourcefile(DVS128Gesture))
    config = dict(dataset='DVS128 Gesture',T=T,split_by='number',resolution=[128,128],
                  official_split=True,split='original trials_to_train.txt and trials_to_test.txt',
                  seed=2026100321,classes=[names[i] for i in range(11)],resources=resources,
                  parser_path=str(parser_path),parser_sha256=digest(parser_path,'sha256'),
                  split_lists_sha256={split:digest(data/f'trials_to_{split}.txt','sha256') for split in splits},
                  subjects=subjects,label_segmentation='Independent original[start,end) intervals; preserve repeated labels',
                  horizontal_flip=False)
    write_json(frames_root/'preparation.json',config)

    def trial(job):
        split,name = job
        events = DVS128Gesture.load_origin_data(str(data/name))
        labels = np.atleast_2d(np.loadtxt(data/(Path(name).stem+'_labels.csv'),delimiter=',',skiprows=1,dtype=np.uint64))
        rows = []
        for index,(label,start,end) in enumerate(labels):
            target = int(label)-1
            if not 0<=target<11 or end<=start:
                raise ValueError('Invalid original gesture label interval')
            mask = (events['t']>=start) & (events['t']<end)
            sample = {key:value[mask] for key,value in events.items()}
            count = len(sample['t'])
            if not count:
                raise ValueError(f'Empty gesture interval: {name}/{index}')
            frames = integrate_events_by_fixed_frames_number(sample,'number',T,128,128)
            if int(frames.sum())!=count or frames.max()>np.iinfo(np.uint32).max:
                raise ValueError('Event loss or overflow in frame integration')
            out = frames_root/split/str(target)/(Path(name).stem+f'_interval{index}.npz')
            out.parent.mkdir(parents=True,exist_ok=True)
            temporary = out.with_suffix('.pending')
            with temporary.open('wb') as f:
                np.savez_compressed(f,frames=frames.astype(np.uint32))
            temporary.replace(out)
            rows.append(dict(path=str(out.relative_to(frames_root)),target=target,label=names[target],split=split,
                             raw_trial=name,csv_row=index,interval_start_usec=int(start),interval_end_usec=int(end),
                             event_count=count,nonmonotonic_timestamps=int(np.count_nonzero(sample['t'][1:]<sample['t'][:-1]))))
        print(f'Prepared {split}/{name}: {len(rows)} original intervals',flush=True)
        return rows

    jobs = [(split,name) for split in ['train','test'] for name in splits[split]]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = [row for trial_rows in pool.map(trial,jobs) for row in trial_rows]
    counts = {split:sum(row['split']==split for row in rows) for split in splits}
    if len({row['path'] for row in rows})!=len(rows):
        raise ValueError('Duplicate cache paths')
    write_json(frames_root/'manifest.json',dict(**config,status='complete',samples=rows,
               train_count=counts['train'],test_count=counts['test']))
    print(f'Complete original official split: {counts}',flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--T',type=int,default=4)
    parser.add_argument('--workers',type=int,default=2)
    args = parser.parse_args()
    if args.T<1 or args.workers<1:
        parser.error('T and workers must be positive')
    try:
        prepare(args.root,args.T,args.workers)
    except Exception as error:
        write_json(args.root/'preparation_error.json',dict(status='failed',error=repr(error)))
        raise
