"""Recompute acceptance contrasts, blind agreement, and label sensitivity."""
from collections import defaultdict
from math import isclose

from common import HERE, dump, load, now, read_rows, sha
from metrics import ci, evaluate_answer

MODELS = ('base', 'control', 'targeted')
SEED = 42


def score_index(split, cap):
    rows = read_rows(HERE/f'data/{split}.jsonl')
    truth = {r['id']:r for r in rows}
    groups = defaultdict(list)
    for row in rows:
        groups[row['pair']].append(row['id'])
    result = {}
    for model in MODELS:
        raw = read_rows(HERE/f'predictions/{split}-{model}-{SEED}.jsonl')
        original = {r['id']:r['prediction'] for r in raw if r['cap']==cap and r['condition']=='original'}
        assert original.keys()==truth.keys()
        result[model] = {key:{'raw':value,**evaluate_answer(value,truth[key])} for key,value in original.items()}
    return truth, groups, result


def compare(groups, result, metric):
    a,b = result['targeted'],result['control']
    if metric=='pair':
        deltas = [int(all(a[i]['semantic'] for i in ids))-int(all(b[i]['semantic'] for i in ids)) for ids in groups]
    else:
        deltas = [sum(int(a[i][metric])-int(b[i][metric]) for i in ids)/len(ids) for ids in groups]
    return {**ci(deltas),'pairs':len(deltas),'questions':sum(len(ids) for ids in groups)}


def main():
    status = load(HERE/'status.json')
    verification = load(HERE/'delivery_verification.json')
    assert status['status']=='complete' and verification['status']=='run_artifacts_verified'
    assert status['protocol_sha256']==verification['protocol_sha256']==sha(HERE/'protocol.json')
    cap = load(HERE/'decode_selection.json')['selected_cap']
    flags = load(HERE/'review_image_flags.json')
    badges = set(flags['ambiguous_natural_photos'])
    assert flags['protocol_sha256']==status['protocol_sha256']
    result = {'checked_at':now(),'protocol_sha256':status['protocol_sha256'],'selected_cap':cap,'splits':{}}
    for split in ('test','challenge'):
        truth, groups, scores = score_index(split,cap)
        published = load(HERE/f'{split}-results-{SEED}-3models.json')
        ab = published['comparisons']['targeted_vs_control']
        contrast = {metric:compare(list(groups.values()),scores,metric) for metric in ('semantic','pair','short')}
        for metric, value in contrast.items():
            assert all(isclose(x,y,abs_tol=1e-9) for x,y in zip(value['ci95_pp'],ab[metric]['ci95_pp']))
            assert isclose(value['delta_pp'],ab[metric]['delta_pp'],abs_tol=1e-9)
        other_cap = next(int(c) for c in published['models']['base'] if int(c)!=cap)
        alt_truth, alt_groups, alt = score_index(split,other_cap)
        assert alt_truth==truth and alt_groups==groups
        changed_labels = {m:sum(v['semantic']!=alt[m][i]['semantic'] for i,v in scores[m].items()) for m in MODELS}
        blind = {}
        reviews = read_rows(HERE/f'review-{split}-{SEED}.jsonl')
        labels = read_rows(HERE/f'blind-labels-{split}-{SEED}.jsonl')
        key = read_rows(HERE/f'review-key-{split}-{SEED}.jsonl')
        assert len(reviews)==len(labels)==len(key)
        assert {r['id'] for r in reviews}==set(flags['samples'][split])
        for r,label,k in zip(reviews,labels,key):
            model,ident = k['model'],r['id']
            assert r['review_id']==label['review_id']==k['review_id'] and label['id']==ident
            assert scores[model][ident]['raw']==r['prediction']
            assert scores[model][ident]['semantic']==k['parser_correct']
            model_summary = blind.setdefault(model,{'n':0,'reference_correct':0,'parser_agreement':0,
                                                      'ambiguous':0,'plausible_alternate':0,
                                                      'repetition':0,'contradiction':0,'extra_hallucination':0})
            model_summary['n'] += 1
            model_summary['reference_correct'] += label['fixed_reference_correct']
            model_summary['parser_agreement'] += (label['fixed_reference_correct']==k['parser_correct'])
            model_summary['ambiguous'] += label['image_reference_ambiguous']
            model_summary['plausible_alternate'] += label['plausible_alternate_to_reference']
            model_summary['repetition'] += label['repetition']
            model_summary['contradiction'] += label['contradictory_answer']
            model_summary['extra_hallucination'] += label['verifiable_extra_hallucination']
        summary = {'recomputed_targeted_vs_control':contrast,'blind':blind,'semantic_changes_at_other_cap':changed_labels}
        if split=='test':
            kept = [ids for ids in groups.values() if badges.isdisjoint(ids)]
            assert len(kept)<len(groups)
            summary['excluding_ambiguous_pairs'] = compare(kept,scores,'semantic')
            text = published['text']
            checks = {
                'semantic':contrast['semantic']['delta_pp']>=5,
                'paired_semantic_ci':contrast['semantic']['ci95_pp'][0]>0,
                'count_and_position':all(compare([ids for ids in groups.values() if truth[ids[0]]['task']==task],
                                                  scores,'semantic')['delta_pp']>0 for task in ('count','position')),
                'both_images_correct':contrast['pair']['delta_pp']>0,
                'text':min(text['targeted']['accuracy']-text[m]['accuracy'] for m in ('base','control'))>=-.02-1e-9,
            }
            assert checks==published['gates']==status['first_seed_gates']
            assert all(checks.values())==status['first_seed_passed']
            summary['gates']=checks
        result['splits'][split]=summary
    result['blind_scope']='One reviewer checked reference correctness, answer contradictions, repetition, and verifiable extra statements before unmasking. Main scores use the fixed rule; flagged image pairs affect sensitivity only.'
    dump(HERE/'acceptance_analysis.json',result)
    print(result)


if __name__=='__main__':
    main()
