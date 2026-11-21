"""Download verified CIFAR10-DVS archives and freeze a reproducible 90/10 split.

Uses the installed SpikingJelly parser and its published resource checksums.
Raw zip files remain available for future T values. No GPU is used.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import time
import zipfile

import numpy as np
import requests


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def digest(path, algorithm='md5'):
    h = hashlib.new(algorithm)
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def download_verified(url, target, expected_md5):
    if target.exists():
        if digest(target) != expected_md5:
            raise RuntimeError(f'Existing archive has invalid checksum: {target}')
        return
    partial = target.with_suffix(target.suffix + '.part')
    for attempt in range(6):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {'Range': f'bytes={offset}-'} if offset else {}
            # Obtain a fresh short-lived redirect rather than a cached signature.
            with requests.get(url, params={'download_prepare': time.time_ns()},
                              headers=headers, stream=True, timeout=(30, 120)) as r:
                r.raise_for_status()
                if offset and r.status_code == 206:
                    if not r.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                        raise RuntimeError('Unexpected HTTP range response')
                    mode = 'ab'
                else:
                    mode = 'wb'
                with partial.open(mode) as f:
                    for chunk in r.iter_content(1024 * 1024):
                        f.write(chunk)
            if digest(partial) != expected_md5:
                # Keep the failed file for diagnosis; never use it as source data.
                partial.rename(partial.with_name(partial.name + f'.invalid.{time.time_ns()}'))
                raise RuntimeError(f'Archive checksum mismatch: {target.name}')
            partial.replace(target)
            print(f'Verified {target.name}: {target.stat().st_size} bytes', flush=True)
            return
        except (requests.RequestException, RuntimeError) as exc:
            print(f'{target.name} attempt {attempt + 1}: {exc}', flush=True)
            if attempt == 5:
                raise
            time.sleep(min(2 ** attempt, 20))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--T', type=int, default=4)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.T < 1 or args.workers < 1:
        parser.error('T and workers must be positive')
    from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
    from spikingjelly.datasets import integrate_events_by_fixed_frames_number

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / '.prepare.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    raw = root / 'download'
    frames_root = root / f'frames_number_{args.T}'
    raw.mkdir(exist_ok=True)
    frames_root.mkdir(exist_ok=True)
    resources = CIFAR10DVS.resource_url_md5()
    parser_path = Path(inspect.getsourcefile(CIFAR10DVS))
    config = dict(dataset='CIFAR10-DVS', T=args.T, split_by='number',
                  resolution=[128, 128], parser_path=str(parser_path),
                  parser_sha256=digest(parser_path, 'sha256'), resources=resources,
                  split='natural numeric filename order; first 900/class train, last 100/class test',
                  official_split=False, seed=2026100321)
    write_json(frames_root / 'preparation.json', config)

    def prepare_class(resource):
        filename, url, md5 = resource
        name = Path(filename).stem
        class_root = frames_root / name
        class_root.mkdir(exist_ok=True)
        status_path = frames_root / f'{name}_status.json'
        target = raw / filename
        write_json(status_path, dict(status='downloading', resource=resource))
        download_verified(url, target, md5)
        records = []
        with zipfile.ZipFile(target) as archive:
            members = [p for p in archive.namelist() if p.endswith('.aedat') and '__MACOSX' not in p]
            members.sort(key=lambda p: int(re.search(r'_(\d+)\.aedat$', p).group(1)))
            if len(members) != 1000:
                raise RuntimeError(f'{name}: expected 1000 samples, found {len(members)}')
            for index, member in enumerate(members):
                # ZIP CRC validation occurs when reading each complete member.
                import io
                from spikingjelly.datasets.cifar10_dvs import load_events
                with archive.open(member) as f:
                    t, x, y, p = load_events(io.BytesIO(f.read()), x_mask=0xfe,
                                            x_shift=1, y_mask=0x7f00, y_shift=8,
                                            polarity_mask=1, polarity_shift=None)
                events = dict(t=t, x=127-y, y=127-x, p=1-p.astype(int))
                if len(t) == 0 or not ((x < 128).all() and (y < 128).all()):
                    raise RuntimeError(f'Invalid event coordinates or empty stream: {member}')
                frames = integrate_events_by_fixed_frames_number(events, 'number', args.T, 128, 128)
                if int(frames.sum()) != len(t) or frames.max() > np.iinfo(np.uint32).max:
                    raise RuntimeError(f'Event count mismatch or overflow: {member}')
                dest = class_root / (Path(member).stem + '.npz')
                tmp = dest.with_suffix('.npz.tmp')
                with tmp.open('wb') as f:
                    np.savez_compressed(f, frames=frames.astype(np.uint32))
                tmp.replace(dest)
                records.append(dict(path=str(dest.relative_to(frames_root)),
                                    source_member=member, label=name,
                                    split='train' if index < 900 else 'test',
                                    event_count=len(t),
                                    nonmonotonic_timestamps=int(np.count_nonzero(t[1:] < t[:-1]))))
                if (index + 1) % 100 == 0:
                    write_json(status_path, dict(status='converting', completed=index + 1,
                                                 archive_md5=md5))
        write_json(status_path, dict(status='complete', completed=1000, archive_md5=md5))
        print(f'Prepared {name}: 900 train / 100 test', flush=True)
        return records

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            grouped = list(pool.map(prepare_class, resources))
        records = [row for group in grouped for row in group]
        labels = sorted({row['label'] for row in records})
        for row in records:
            row['target'] = labels.index(row['label'])
        manifest = dict(**config, status='complete', classes=labels,
                        train_count=9000, test_count=1000, samples=records)
        write_json(frames_root / 'manifest.json', manifest)
        print('Complete: 9000 train / 1000 held-out test; no official dataset split.', flush=True)
    except Exception as exc:
        write_json(frames_root / 'error.json', dict(status='failed', error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
