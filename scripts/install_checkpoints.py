"""Verify and install project checkpoints downloaded from the private release."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('asset_dir',type=Path)
    args=parser.parse_args()
    manifest=json.loads((ROOT/'evidence/checkpoints.json').read_bytes())
    files=[]
    for item in manifest['assets']:
        source=args.asset_dir/item['filename']
        if not source.is_file():
            continue
        if source.stat().st_size!=item['bytes'] or digest(source)!=item['sha256']:
            raise SystemExit(f'Checkpoint size or SHA-256 mismatch: {source.name}')
        files.append((source,ROOT/item['install_path'],item['sha256']))
    if not files:
        raise SystemExit('No listed checkpoint assets found.')
    for source, destination, expected in files:
        if destination.exists():
            if digest(destination)!=expected:
                raise SystemExit(f'Existing checkpoint differs: {destination.relative_to(ROOT)}')
        else:
            destination.parent.mkdir(parents=True,exist_ok=True)
            temporary=destination.with_suffix(destination.suffix+'.part')
            shutil.copyfile(source,temporary)
            temporary.replace(destination)
        print(destination.relative_to(ROOT))


if __name__=='__main__':
    main()
