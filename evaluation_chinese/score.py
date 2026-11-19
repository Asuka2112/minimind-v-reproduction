"""Score frozen answer aliases and produce paired comparisons and a review file."""
import collections
import random
import re
import unicodedata
from datetime import datetime, timezone
import numpy as np
from prepare import HERE, MODELS, read_rows, sha, write


def normalize(text):
    text = unicodedata.normalize('NFKC', text).strip()
    text = re.sub(r'^(?:答\s*[:：]|答案\s*(?:[:：]|是))\s*', '', text, count=1)
    text = re.sub(r'^[是为]\s*', '', text, count=1)
    text = text.strip(' \n\t。.!！?？;；:：,，"“”\'‘’')
    return re.sub(r'\s+', '', text).casefold()


def main(protocol, records, started_at):
    questions = read_rows(HERE / 'data/questions.jsonl')
    gold = {row['id']: row for row in questions}
    protocol_sha = sha(HERE / 'protocol.json')
    index = {}
    for record in records:
        r = gold[record['id']]
        assert record['protocol_sha256'] == protocol_sha
        expected_image = {'correct': r['image'], 'swapped': r['swapped_image'], 'none': None}[record['condition']]
        assert record['image'] == expected_image
        assert 0 < len(record['generated_ids']) <= protocol['decode']['max_new_tokens']
        k = (record['model'], record['prompt'], record['condition'], record['id'])
        assert k not in index
        prediction = normalize(record['prediction'])
        original_aliases = {normalize(a) for a in r['aliases']}
        displayed_aliases = ({normalize(a) for a in r['swapped_aliases']}
                             if record['condition'] == 'swapped' else original_aliases)
        index[k] = dict(record, correct=prediction in original_aliases,
                        displayed_correct=prediction in displayed_aliases)
    assert len(index) == len(gold)*len(MODELS)*len(protocol['conditions'])*len(protocol['prompts'])

    def get(model, prompt, condition, rows):
        return [index[model, prompt, condition, row['id']] for row in rows]

    def rate(predictions, key):
        return float(np.mean([r[key] for r in predictions]))

    def paired(a, b, rows):
        by_pair = collections.defaultdict(list)
        for x, y, r in zip(a, b, rows):
            by_pair[r['pair']].append(int(x['correct'])-int(y['correct']))
        blocks = np.array([np.mean(block) for block in by_pair.values()])
        rng = np.random.default_rng(1726)
        sampling = rng.choice(blocks, size=(2000, len(blocks)), replace=True).mean(axis=1)
        return {'difference_pp': float(blocks.mean()*100), 'ci95_pp': list(np.percentile(sampling*100, [2.5, 97.5])),
                'pairs': len(blocks)}

    subsets = {
        'synthetic': [r for r in questions if r['source']=='synthetic'],
        'real_and_mirrored': [r for r in questions if r['source']!='synthetic'],
        'real_unmodified': [r for r in questions if r['source']=='natural' or
                            (r['source']=='natural_mirror' and not r['provenance']['mirror'])],
    }
    scores = {}
    for group, rows in subsets.items():
        results = {}
        for model in MODELS:
            for prompt in protocol['prompts']:
                conditions = {}
                for condition in protocol['conditions']:
                    pred = get(model, prompt, condition, rows)
                    conditions[condition] = {
                        'normalized_accuracy': rate(pred, 'correct'),
                        'displayed_image_accuracy': rate(pred, 'displayed_correct') if condition != 'none' else None,
                        'length_cap_rate': rate(pred, 'hit_length_cap'),
                        'by_category': {category: rate([p for p, r in zip(pred, rows) if r['category']==category], 'correct')
                                        for category in sorted({r['category'] for r in rows})},
                    }
                originals = get(model, prompt, 'correct', rows)
                swaps = get(model, prompt, 'swapped', rows)
                paired_indices = collections.defaultdict(list)
                for idx, row in enumerate(rows):
                    paired_indices[row['pair']].append(idx)
                both_correct = [all(originals[i]['correct'] for i in ids) for ids in paired_indices.values() if len(ids)==2]
                results[f'{model}/{prompt}'] = {
                    'conditions': conditions,
                    'answer_changes_on_swap': float(np.mean([normalize(x['prediction']) != normalize(y['prediction'])
                                                               for x, y in zip(originals, swaps)])),
                    'both_images_correct': float(np.mean(both_correct)),
                    'both_images_correct_pairs': len(both_correct),
                    'correct_minus_swapped': paired(originals, swaps, rows),
                    'correct_minus_none': paired(originals, get(model, prompt, 'none', rows), rows),
                }
        results['prompt_effect_on_original'] = {
            m: paired(get(m, 'short', 'correct', rows), get(m, 'baseline', 'correct', rows), rows) for m in MODELS
        }
        results['reproduced_minus_official_on_original'] = {
            p: paired(get('reproduced', p, 'correct', rows), get('official', p, 'correct', rows), rows) for p in protocol['prompts']
        }
        scores[group] = {'n': len(rows), **results}

    # Unmodified photo responses in all conditions, blinded to model and prompt.
    review = []
    for record in index.values():
        r = gold[record['id']]
        if r['source'] != 'synthetic' and not r['provenance'].get('mirror', False):
            ref = r['swapped_answer'] if record['condition']=='swapped' else r['answer']
            review.append({'id': r['id'], 'shown_image': record['image'], 'question': r['question'],
                           'reference': ref, 'prediction': record['prediction'], 'model': record['model'],
                           'prompt': record['prompt'], 'condition': record['condition']})
    random.Random(1726).shuffle(review)
    key = []
    for i, row in enumerate(review, 1):
        row['review_id'] = i
        key.append({'review_id': i, **{k: row.pop(k) for k in ('model', 'prompt', 'condition')}})
    write(HERE/'review.json', review)
    write(HERE/'review_key.json', key)
    report = {
        'name': '评测', 'finished_at': datetime.now(timezone.utc).isoformat(), 'started_at': started_at,
        'protocol_sha256': protocol_sha, 'prediction_count': len(index), 'scores': scores,
        'review_items': len(review), 'review_status': 'unrated',
        'prediction_sha256': {p.name: sha(p) for p in sorted((HERE/'predictions').glob('*.jsonl'))},
    }
    write(HERE/'results.json', report)
    lines = [
        '# 中文图片问答评测', '',
        '152 题，76 个同题异答案图片对。官方权重和本次复现权重使用相同的 FP32 推理、192 token 贪心解码。',
        '第一种提示直接提问，第二种提示追加简短作答要求。换图保留题目，通过同题下另一张图检查答案是否跟随图片。',
        '', '主指标使用冻结的标准答案与同义词做归一化精确匹配。review.json 保存真实图片回答供语义复核。',
        '真实图片沿用此前外部评测图源并重写题目。真实子集采用定向选样，可控图像由程序生成。', '',
        '| 子集 | 模型/提示 | 原图正确率 | 换图对原图答案 | 换图对新图答案 | 无图正确率 | 原图长度上限 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for group, block in scores.items():
        for name, result in block.items():
            if '/' not in name:
                continue
            c = result['conditions']
            lines.append(f"| {group} ({block['n']}) | {name} | {c['correct']['normalized_accuracy']:.1%} | "
                         f"{c['swapped']['normalized_accuracy']:.1%} | {c['swapped']['displayed_image_accuracy']:.1%} | "
                         f"{c['none']['normalized_accuracy']:.1%} | {c['correct']['length_cap_rate']:.1%} |")
    lines += ['', 'results.json 附带逐类别分数、换图答案变化比例、两张图都答对的比例及配对区间。', '']
    (HERE/'REPORT.md').write_text('\n'.join(lines))
    return report
