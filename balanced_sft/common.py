"""Shared provenance and file utilities for the controlled continuation experiment."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os

HERE = Path(__file__).resolve().parent
PROJECTS = Path(os.environ.get("MINIMIND_V_WORKSPACE", HERE.parent / "workspace")).expanduser().resolve()
ORIGINAL = PROJECTS / 'minimind-v-official-20261102'
SOURCE = ORIGINAL / 'src'
OLD_INDEX = PROJECTS / 'vlm-baseline-20261026/data/image_index.sqlite3'
OLDER_EVAL = PROJECTS / 'minimind-v-independent-eval-20261103'
PRIOR = Path(os.environ.get("MINIMIND_V_PRIOR", HERE.parent / "targeted_sft")).expanduser().resolve()
BASE = PRIOR / 'weights/targeted-42.pth'
BASE_SHA = '0516e8103022d413a3f36b15a1c0c5bb2f88f7e24c31963c96a6d7507b5380b5'
OFFICIAL = ORIGINAL / 'weights/official_sft_vlm_768.pth'
OFFICIAL_SHA = '0d3fee0970c2174d22226de25d273e2f375bda8ed39b8f1673e1519c94264f9e'
CORPUS = ORIGINAL / 'data/sft_i2t.parquet'
CORPUS_SHA = '712f4026cd0e21b369feddca7334b1e465cb8182b5f298006f3f4f877f926643'
SEED = 1726
COCO = {
    'val0.parquet': 'dc7a465827eab75e825eb38c9993e6b04fcf918fc2f034504d5760aedfa71d7c',
    'val1.parquet': 'fa7fcf60b688c5555b92778ca9654da9400c72d1612b09f3a094a18266e8d859',
    'instances.json': 'e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f',
}

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def load(path):
    return json.loads(Path(path).read_text())

def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream]

def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)

def dump_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    tmp.replace(path)

def now():
    return datetime.now(timezone.utc).isoformat()

def image_fingerprint(image):
    image = image.convert('RGB')
    rgb = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
    pixels = list(image.resize((9, 8)).convert('L').getdata())
    dhash = sum(int(pixels[y*9+x] > pixels[y*9+x+1]) << (y*8+x) for y in range(8) for x in range(8))
    return rgb, dhash
