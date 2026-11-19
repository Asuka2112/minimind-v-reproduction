"""Read-only evaluation of the completed official MiniMind-V reproduction."""
import datetime
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN = Path('/workspace/projects/minimind-v-official-20261102')
BENCH = Path('/workspace/projects/vlm-format-validation-20261030')
OLD = Path('/workspace/projects/vlm-baseline-20261026')
PYTHON = '/workspace/projects/minimind3-reproduction-20261019/env311/bin/python'
MODELS = {
    'initial': TRAIN / 'weights/llm_768.pth',
    'pretrain': TRAIN / 'runs/pretrain/out/pretrain_vlm_768.pth',
    'sft': TRAIN / 'runs/sft/out/sft_vlm_768.pth',
    'official': TRAIN / 'weights/official_sft_vlm_768.pth',
}

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def sha(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)

def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def strict_choice(text):
    import re
    match = re.fullmatch(r'\s*\(?([ABCDabcd])\)?[.。]?\s*', text)
    return match.group(1).upper() if match else None

def official_choice(text):
    # Exact priority and normalization of pinned MMStar_eval, without a judge.
    text = text.lower().strip().replace('\n', ' ')
    for prefix in ['', '(', 'option ', 'the answer is ']:
        if text.startswith(prefix) and len(text) > len(prefix):
            letter = text[len(prefix)]
            if letter in 'abcd':
                return letter.upper()
    return None
