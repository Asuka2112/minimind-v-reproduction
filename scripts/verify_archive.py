"""Check archived bytes and optionally replay both matched SFT comparisons."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_bytes())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def archive_bytes(root, relative, inventory):
    item = inventory.get(str(relative))
    return (gzip.decompress((root / item['archive']).read_bytes()) if item
            else (root / relative).read_bytes())


def verify(root, inventory):
    for line in (root / 'evidence/files.sha256').read_text(encoding='utf-8').splitlines():
        expected, relative = line.split('  ', 1)
        assert digest((root / relative).read_bytes()) == expected, relative

    for target, item in inventory.items():
        compressed = (root / item['archive']).read_bytes()
        assert digest(compressed) == item['compressed_sha256'], item['archive']
        raw = gzip.decompress(compressed)
        assert len(raw) == item['expanded_bytes'] and digest(raw) == item['expanded_sha256'], target

    for item in load(root / 'evidence/path-sanitization.json'):
        rel = item['path']
        assert digest(archive_bytes(root, rel, inventory)) == item['archived_sha256'], rel

    for experiment, expected in (('evaluation_independent',18256), ('evaluation_chinese',1824),
                                 ('targeted_sft',7120), ('balanced_sft',9420)):
        folder = root / experiment / 'predictions'
        streams = set(p.relative_to(root).as_posix() for p in folder.glob('*.jsonl'))
        streams.update(p.relative_to(root).as_posix()[:-3] for p in folder.glob('*.jsonl.gz'))
        n = sum(len(archive_bytes(root, stream, inventory).splitlines()) for stream in streams)
        assert n == expected, (experiment,n,expected)
        print(f'{experiment}: {n:,} prediction records')

    for experiment in ('targeted_sft','balanced_sft'):
        manifest = load(root / experiment / 'data/manifest.json')
        for filename, expected in manifest['files'].items():
            rel = Path(experiment) / 'data' / filename
            assert digest(archive_bytes(root, rel, inventory)) == expected, rel

    for path in sorted((root / 'reports').glob('*.pdf')):
        assert path.read_bytes().startswith(b'%PDF-'), path


def expand(root, inventory):
    for relative, item in inventory.items():
        output = root / relative
        if output.exists():
            if digest(output.read_bytes()) != item['expanded_sha256']:
                raise ValueError(f'Existing file differs from archive: {relative}')
            continue
        payload = archive_bytes(root, relative, inventory)
        if digest(payload) != item['expanded_sha256']:
            raise ValueError(f'Invalid expanded digest: {relative}')
        output.write_bytes(payload)


def replay(inventory):
    experiment_results = {'targeted_sft':'acceptance_summary.json',
                          'balanced_sft':'acceptance_analysis.json'}
    with TemporaryDirectory(prefix='vqa-replay-') as tmp:
        working = Path(tmp)
        for experiment in experiment_results:
            shutil.copytree(ROOT / experiment, working / experiment,
                            ignore=shutil.ignore_patterns('__pycache__','*.pth','*.part','images','coco','logs'))
        expand(working,{p:m for p,m in inventory.items() if p.split('/')[0] in experiment_results})
        env=os.environ.copy()
        env.update(MINIMIND_V_WORKSPACE=str(working/'workspace'), MINIMIND_V_PRIOR=str(working/'targeted_sft'))
        for experiment, result_file in experiment_results.items():
            subprocess.run([sys.executable,'analyze_acceptance.py'],cwd=working/experiment,env=env,
                           stdout=subprocess.DEVNULL,check=True)
            actual=load(working/experiment/result_file)
            published=load(ROOT/experiment/result_file)
            for result in (actual,published):
                result.pop('checked_at',None)
            assert actual==published, experiment
            print(f'{experiment}: replay matches archived conclusions')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expand',action='store_true',help='materialize the byte-checked JSONL streams')
    parser.add_argument('--replay',action='store_true',help='recompute both A/B reports in an isolated directory')
    args=parser.parse_args()
    inventory=load(ROOT/'evidence/streams.json')
    verify(ROOT,inventory)
    if args.expand:
        expand(ROOT,inventory)
    if args.replay:
        replay(inventory)
    print(f'Archive verified: {len(inventory)} checked compressed streams.')


if __name__=='__main__':
    main()
