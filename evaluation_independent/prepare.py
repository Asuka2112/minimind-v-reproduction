"""Freeze the complete protocol and input identities BEFORE new model predictions."""
import collections
import json
import random
import shutil
from PIL import Image
import pyarrow.parquet as pq
from common import *

def main():
    assert not (HERE / 'protocol.json').exists(), 'Protocol already frozen; do not overwrite'
    assert not list((HERE / 'predictions').glob('*.jsonl')), 'Predictions already exist'
    (HERE / 'data').mkdir(exist_ok=True)
    source = HERE / 'data/mmstar.parquet'
    assert sha(source) == '29afd74b0134cfab083a8909b5358577ab18fd41c1e612031577cfb3635531c2'
    original = pq.read_table(source).to_pylist()
    prepared = rows(BENCH / 'data/mmstar.jsonl')
    assert len(original) == len(prepared) == 1500
    assert all(x['answer'] in 'ABCD' and len(x['answer']) == 1 for x in original)
    for x, y in zip(original, prepared):
        for k in ['index', 'question', 'answer', 'category', 'l2_category', 'meta_info']:
            assert x[k] == y[k], (x['index'], k)
        path = BENCH / y['image_file']
        assert sha(path) == y['image_file_sha256']
        with Image.open(path) as im:
            im = im.convert('RGB')
            import hashlib
            assert hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest() == y['image_sha256']
        raw = x['image'] if isinstance(x['image'], bytes) else x['image']['bytes']
        assert raw == path.read_bytes()
    # Do not reuse any prior project split or prior predictions. Newly fixed swap map.
    images = sorted({x['image_sha256'] for x in prepared})
    random.Random(20261103).shuffle(images)
    swaps = {a: images[(i + 1) % len(images)] for i, a in enumerate(images)}
    assert all(a != b for a, b in swaps.items())
    image_paths = {x['image_sha256']: str(BENCH / x['image_file']) for x in prepared}
    clean = [{k: x[k] for k in ['id', 'index', 'question', 'answer', 'category', 'l2_category', 'meta_info', 'image_sha256', 'image_file_sha256', 'dhash']} |
             {'image_file': image_paths[x['image_sha256']], 'shuffled_image_sha256': swaps[x['image_sha256']],
              'shuffled_image_file': image_paths[swaps[x['image_sha256']]]} for x in prepared]
    (HERE / 'data/mmstar.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in clean))
    text = []
    rng = random.Random(20261103)
    for op, count in [('+', 24), ('-', 24), ('×', 16)]:
        seen = set()
        while len(seen) < count:
            a, b = (rng.randint(11, 89), rng.randint(1, 9)) if op != '×' else (rng.randint(2, 9), rng.randint(2, 9))
            if (a, b) in seen: continue
            seen.add((a, b))
            answer = a + b if op == '+' else a - b if op == '-' else a * b
            text.append({'id': f'text-{len(text):03d}', 'operation': op, 'question': f'请计算{a}{op}{b}，只输出计算结果的阿拉伯数字。', 'answer': str(answer)})
    (HERE / 'data/text64.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in text))
    protocol = {
        'frozen_at': now(), 'status': 'frozen_before_new_model_predictions',
        'purpose': 'Independent external evaluation of completed official reproduction; no training or test-driven prompt/model selection.',
        'source_commit': '041d528e037a8a373c3c8aa7278a3ba3bd26b6dc',
        'dataset': {'name': 'Lin-Chen/MMStar', 'revision': 'bc98d668301da7b14f648724866e57302778ab27',
                    'source_sha256': sha(source), 'rows': 1500, 'unique_rgb_images': len(images),
                    'categories': dict(collections.Counter(x['category'] for x in clean))},
        'models': {k: {'path': str(v), 'sha256': sha(v)} for k, v in MODELS.items()},
        'precision': 'FP16 native checkpoint artifacts reloaded into FP32; FP32 vision and LM, TF32 disabled, no autocast',
        'seed': 42, 'image_processing': 'same pinned SigLIP2 processor, 256x256, 64 visual tokens',
        'conditions': ['correct', 'shuffled', 'none'],
        'swap': 'seed20261103 permutation of unique RGB hashes followed by a one-position rotation; every image replaced by a different RGB image',
        'prompt_suffix': "\nAnswer with the option's letter from the given choices directly, such as answer letter 'A' only.",
        'chat': 'upstream tokenizer apply_chat_template, add_generation_prompt=True, open_thinking=False; image_pad x64 only in image conditions',
        'decode': {'method': 'greedy', 'max_new_tokens': 16, 'temperature': 1.0, 'top_p': 1.0, 'top_k': 0, 'repetition_penalty': 1.0},
        'primary': 'Full1500 strict single-letter generation accuracy; invalid/empty/truncated nonletter answers count wrong; no dropping failed answers',
        'secondary': ['pinned official MMStar prefix-rule generation accuracy', 'strict-format compliance and length-cap rate',
                      'argmax first-position logits restricted to A/B/C/D (all must be single tokens); diagnostic, not generation or leaderboard accuracy',
                      'six category accuracies and answer-letter distribution',
                      'correct-image minus shuffled/no-image paired differences for each model'],
        'comparisons': ['sft minus initial on correct images', 'sft minus official on correct images', 'sft correct minus shuffled', 'sft correct minus none'],
        'uncertainty': '2000 paired bootstrap replicates, seed20261103; clusters are exact RGB image hashes, all questions for each sampled image retained; percentile95% CI; descriptive, no multiplicity-corrected significance claims',
        'overlap': 'Audit BOTH complete official training files. Reuse full-SFT raw-image index only after raw SHA coverage verification; decode all absent or previously invalid images. Exact raw bytes, exact decoded RGB, and direct dHash64 distance<=3 conservative candidates. Report full benchmark and subset excluding exact/near train matches; never claim foundation-model contamination ruled out.',
        'text_diagnostic': '64 newly constructed fixed Chinese arithmetic prompts (24add/24subtract/16multiply), greedy16, exact numeral and answer-token NLL without EOS. A narrow retention diagnostic, not a general language benchmark; no capability threshold selected after results.',
        'limitations': ['English difficult benchmark differs from Chinese caption training and small65M decoder capacity.',
                        'One training seed; bootstrap measures test-image sampling uncertainty only.',
                        'Official reference training provenance/overlap may differ; no fair-training-compute claim.',
                        'Earlier project used same public benchmark; this run fixes new protocol without using earlier model results. Benchmark public training contamination cannot be excluded.'],
        'inputs': {str(p.relative_to(HERE)): sha(p) for p in [HERE / 'data/mmstar.jsonl', HERE / 'data/text64.jsonl']},
        'code': {p.name: sha(p) for p in sorted(HERE.glob('*.py'))},
        'source_files': {str(p): sha(p) for p in sorted((TRAIN / 'src/model').glob('*')) if p.is_file() and p.suffix in ['.py', '.json']},
        'vision_files': {str(p): sha(p) for p in sorted((TRAIN / 'src/model/siglip2-base-p32-256-ve').glob('*')) if p.is_file()},
    }
    dump(HERE / 'protocol.json', protocol)
    print('PROTOCOL_FROZEN', sha(HERE / 'protocol.json'), flush=True)

if __name__ == '__main__': main()
