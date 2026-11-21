"""Extract the Stanford Tiny-ImageNet archive and validate its official splits."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.tiny_imagenet import dataset_manifest, sha256


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.archive.with_suffix(a.archive.suffix+'.aria2').exists():
        raise RuntimeError('Download is still incomplete')
    destination = a.root/'tiny-imagenet-200'
    if not destination.exists():
        temporary = a.root/'.tiny-extracting'
        if temporary.exists(): raise RuntimeError('Incomplete prior extraction retained: '+str(temporary))
        temporary.mkdir(parents=True)
        with zipfile.ZipFile(a.archive) as z:
            for info in z.infolist():
                path = Path(info.filename)
                if path.is_absolute() or '..' in path.parts or path.parts[0] != 'tiny-imagenet-200':
                    raise ValueError('Unexpected archive member')
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Archive symlink is not expected')
            if z.testzip() is not None: raise ValueError('ZIP CRC check failed')
            z.extractall(temporary)
        os.replace(temporary/'tiny-imagenet-200', destination)
        temporary.rmdir()
    manifest = dataset_manifest(destination)
    a.output.mkdir(parents=True, exist_ok=True)
    path = a.output/'manifest.json'
    path.write_text(json.dumps(manifest, indent=2)+'\n')
    summary = {k: manifest[k] for k in ('dataset','root','image_size','training_images','validation_images','manifest_sha256','validation_labels')}
    summary.update(status='ready', archive=str(a.archive.resolve()), archive_sha256=sha256(a.archive),
                   archive_bytes=a.archive.stat().st_size, checked_at=time.time(),
                   manifest_path=str(path.resolve()), official_source='https://cs231n.stanford.edu/tiny-imagenet-200.zip')
    (a.output/'dataset_ready.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary))


if __name__ == '__main__': main()
