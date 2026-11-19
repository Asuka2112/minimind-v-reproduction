"""Verify the transferred run and summarize the blinded semantic review."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import median

import numpy as np
from transformers import AutoTokenizer
from score import normalize

ROOT = Path(__file__).resolve().parent


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads((ROOT / path).read_text())


def rows(path):
    return [json.loads(line) for line in (ROOT / path).read_text().splitlines()]


def paired(left, right, reviewed):
    differences = defaultdict(list)
    for item in reviewed:
        question = item['id']
        differences[item['pair']].append(int(left[question]) - int(right[question]))
    blocks = list(differences.values())
    totals = np.array([sum(block) for block in blocks])
    sizes = np.array([len(block) for block in blocks])
    rng = np.random.default_rng(1726)
    draws = rng.integers(len(blocks), size=(2000, len(blocks)))
    draws_pp = 100 * totals[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    return {'n': int(sizes.sum()), 'pairs': len(blocks),
            'difference_pp': float(100 * totals.sum() / sizes.sum()),
            'ci95_pp': np.percentile(draws_pp, [2.5, 97.5]).tolist(),
            'improved': sum(left[q] and not right[q] for q in left),
            'worsened': sum(right[q] and not left[q] for q in left)}


def main():
    state = read('status.json')
    report = read('results.json')
    protocol = read('protocol.json')
    protocol_sha = digest(ROOT/'protocol.json')
    assert state['status'] == 'complete' and state['name'] == '评测'
    assert state['results_sha256'] == digest(ROOT/'results.json')
    assert state['protocol_sha256'] == report['protocol_sha256'] == protocol_sha
    for name, sha256 in protocol['code_sha256'].items():
        assert digest(ROOT/name) == sha256, name
    assert protocol['dataset_sha256'] == digest(ROOT/'data/questions.jsonl')
    questions = rows('data/questions.jsonl')
    assert protocol['rows'] == len(questions) == 152
    by_id = {r['id']: r for r in questions}
    by_image = {r['image']: r for r in questions}
    assert len(by_id) == len(by_image) == len(questions)
    assert Counter(r['pair'] for r in questions) == dict.fromkeys({r['pair'] for r in questions}, 2)
    for question in questions:
        partner = by_image[question['swapped_image']]
        assert question['pair'] == partner['pair'] and partner['image'] != question['image']
        assert question['question'] == partner['question']
        assert question['swapped_answer'] == partner['answer'] != question['answer']
        assert partner['swapped_image'] == question['image']
        assert digest(ROOT/'data'/question['image']) == question['image_sha256']

    tokenizer_path = ROOT.parent/'model'
    for name in ['tokenizer.json','tokenizer_config.json']:
        assert digest(tokenizer_path/name) == protocol['model_source_sha256'][name]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    vocab_size = len(tokenizer)
    eos = tokenizer.eos_token_id
    data = {}
    for model in protocol['models']:
        path = f'predictions/{model}.jsonl'
        receipt = read(f'receipt_{model}.json')
        predictions = rows(path)
        assert state['worker_exit_codes'][model] == 0
        assert len(predictions) == receipt['rows'] == 912
        assert receipt['protocol_sha256'] == protocol_sha
        assert receipt['checkpoint_sha256'] == protocol['models'][model]['checkpoint_sha256']
        assert digest(ROOT/path) == receipt['sha256'] == report['prediction_sha256'][Path(path).name]
        decoded = tokenizer.batch_decode([p['generated_ids'] for p in predictions], skip_special_tokens=True)
        for p, decoded_text in zip(predictions, decoded):
            tokens = p['generated_ids']
            assert p['prediction'] == decoded_text and p['protocol_sha256'] == protocol_sha and p['model'] == model
            assert 1 <= len(tokens) <= protocol['decode']['max_new_tokens']
            assert all(0 <= tok < vocab_size for tok in tokens)
            assert eos not in tokens[:-1]
            assert p['hit_length_cap'] == (len(tokens) == protocol['decode']['max_new_tokens'] and tokens[-1] != eos)
            q = by_id[p['id']]
            assert p['image'] == {'correct': q['image'], 'swapped': q['swapped_image'], 'none': None}[p['condition']]
            index = (model, p['prompt'], p['condition'], p['id'])
            assert index not in data
            data[index] = p
    assert len(data) == report['prediction_count'] == state['predictions'] == 1824
    assert set(data) == {(m,p,c,r['id']) for m in protocol['models'] for p in protocol['prompts']
                         for c in protocol['conditions'] for r in questions}

    subsets = {'synthetic': [q for q in questions if q['source']=='synthetic'],
               'real_and_mirrored': [q for q in questions if q['source']!='synthetic'],
               'real_unmodified': [q for q in questions if q['source']!='synthetic' and not q['provenance']['mirror']]}
    swap_mismatches = []
    exact_correct = 0
    stats = {}
    for subset, selected in subsets.items():
        stats[subset] = {}
        for model in protocol['models']:
            for prompt in protocol['prompts']:
                group = {}
                for condition in protocol['conditions']:
                    original_score = displayed_score = caps = 0
                    for q in selected:
                        p = data[model,prompt,condition,q['id']]
                        normalized = normalize(p['prediction'])
                        original_score += normalized in {normalize(s) for s in q['aliases']}
                        ref = q['swapped_aliases'] if condition == 'swapped' else q['aliases']
                        displayed_score += normalized in {normalize(s) for s in ref}
                        caps += p['hit_length_cap']
                        if subset != 'real_unmodified' and condition == 'swapped':
                            partner = by_image[q['swapped_image']]
                            other = data[model,prompt,'correct',partner['id']]
                            if p['generated_ids'] != other['generated_ids']:
                                swap_mismatches.append((model, prompt, q['id']))
                    stored = report['scores'][subset][f'{model}/{prompt}']['conditions'][condition]
                    assert stored['normalized_accuracy'] == original_score / len(selected)
                    assert stored['length_cap_rate'] == caps / len(selected)
                    if condition != 'none':
                        assert stored['displayed_image_accuracy'] == displayed_score / len(selected)
                    if subset != 'real_unmodified':
                        exact_correct += original_score
                    group[condition] = {'n': len(selected), 'exact_correct': original_score,
                                        'displayed_correct': displayed_score if condition != 'none' else None,
                                        'length_cap': caps,
                                        'answers_within_10_chars': sum(len(data[model,prompt,condition,q['id']]['prediction'].strip()) <= 10
                                                                       for q in selected),
                                        'median_generated_tokens': median(len(data[model,prompt,condition,q['id']]['generated_ids'])
                                                                          for q in selected)}
                stats[subset][f'{model}/{prompt}'] = group
    assert not swap_mismatches, swap_mismatches[:5]

    review = {r['review_id']: r for r in read('review.json')}
    review_key = {r['review_id']: r for r in read('review_key.json')}
    annotations_path = 'semantic_review.json'
    annotations = read(annotations_path)['annotations']
    expected = {i for i,k in review_key.items() if k['condition']=='correct'}
    assert len(annotations) == 84 and {a['review_id'] for a in annotations} == expected
    verdicts = defaultdict(dict)
    verdict_counts = defaultdict(Counter)
    for annotation in annotations:
        review_id = annotation['review_id']
        key = review_key[review_id]
        item = review[review_id]
        assert item['id'] == annotation['id'] and item['reference'] == by_id[item['id']]['answer']
        m, p = key['model'], key['prompt']
        assert item['prediction'] == data[m,p,'correct',item['id']]['prediction']
        verdict = annotation['verdict']
        verdicts[m,p][item['id']] = verdict == 'correct'
        verdict_counts[m,p][verdict] += 1
    reviewed = subsets['real_unmodified']
    pair_ids = defaultdict(list)
    for q in reviewed:
        pair_ids[q['pair']].append(q['id'])
    complete_review_pairs = [ids for ids in pair_ids.values() if len(ids)==2]
    semantic = {}
    for group, flags in verdicts.items():
        assert len(flags) == len(reviewed)
        correct = sum(flags.values())
        semantic['/'.join(group)] = {'correct': correct, 'n': len(flags),
            'accuracy': correct/len(flags), 'verdict_counts': dict(verdict_counts[group]),
            'both_correct': sum(all(flags[i] for i in ids) for ids in complete_review_pairs),
            'complete_pairs': len(complete_review_pairs),
            'by_category': {c: {'correct': sum(flags[q['id']] for q in reviewed if q['category']==c),
                                'n': sum(q['category']==c for q in reviewed)} for c in sorted({q['category'] for q in reviewed})}}

    comparisons = {f'{m}/short_minus_baseline': paired(verdicts[m,'short'], verdicts[m,'baseline'], reviewed)
                   for m in protocol['models']}
    comparisons['baseline/reproduced_minus_official'] = paired(verdicts['reproduced','baseline'], verdicts['official','baseline'], reviewed)
    comparisons['short/reproduced_minus_official'] = paired(verdicts['reproduced','short'], verdicts['official','short'], reviewed)
    out = {'status': 'verified', 'at': datetime.now(timezone.utc).isoformat(),
           'protocol_sha256': protocol_sha, 'results_sha256': digest(ROOT/'results.json'),
           'question_count': len(questions), 'prediction_count': len(data), 'decoded_locally': len(data),
           'swapped_partner_exact_token_match': len(questions)*len(protocol['models'])*len(protocol['prompts']),
           'exact_correct_count_all_records': exact_correct, 'exact_matching': stats,
           'semantic_review_sha256': digest(ROOT/annotations_path), 'semantic_review': semantic,
           'semantic_paired_comparisons': comparisons}
    (ROOT/'acceptance.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':out['status'], 'records': len(data), 'strict_correct':exact_correct,
                      'semantic': semantic, 'comparisons': comparisons},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
