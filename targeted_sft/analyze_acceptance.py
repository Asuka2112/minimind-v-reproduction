"""Recompute exploratory contrasts and join the frozen blind-review labels."""
from collections import defaultdict
from common import HERE, dump, load, read_rows, sha
from metrics import ci, evaluate_answer

MODELS=('base','official','control','targeted')
METRICS=('semantic','contradiction','repeated_description','clear_visible_hallucination')

def main():
    test=read_rows(HERE/'data/test.jsonl')
    identities={q['id']:q for q in test}
    pairs=defaultdict(list)
    for q in test:
        pairs[q['pair']].append(q['id'])
    assert len(pairs)==120 and all(len(ids)==2 for ids in pairs.values())
    flags=load(HERE/'audit/test_label_review.json')['flags']
    excluded={identities[q]['pair'] for q in flags}
    passed_pairs=[ids for name,ids in pairs.items() if name not in excluded]
    predictions={}
    for m in MODELS:
        rows=[r for r in read_rows(HERE/f'predictions/test-{m}-42.jsonl') if r['condition']=='original' and r['cap']==192]
        predictions[m]={r['id']:evaluate_answer(r['prediction'],identities[r['id']]) for r in rows}
        assert set(predictions[m])==set(identities)

    def contrast(left,right,pool,kind):
        a,b=predictions[left],predictions[right]
        if kind=='pair':
            d=[int(all(a[i]['semantic'] for i in ids))-int(all(b[i]['semantic'] for i in ids)) for ids in pool]
        else:
            d=[sum(int(a[i][kind])-int(b[i][kind]) for i in ids)/len(ids) for ids in pool]
        return {**ci(d),'pairs':len(pool),'questions':sum(map(len,pool))}
    full=list(pairs.values())
    acceptance={
        'format_scope':'Frozen normalized answer parser, 192 token original graph. Raw correctness verified on full dataset; blind audit is a separate diagnostic.',
        'frozen_result_sha256':sha(HERE/'test-results-42.json'),
        'contrasts':{f'{a}_vs_{b}':{kind:contrast(a,b,full,kind) for kind in ('semantic','short','pair')}
                     for a,b in [('targeted','control'),('targeted','base'),('control','base')]},
        'targeted_vs_control_by_task':{t:contrast('targeted','control',[i for i in full if identities[i[0]]['task']==t],'semantic')
                                       for t in ('object','color','count','position')},
        'flag_sensitivity':{'removed_pairs':sorted(excluded),'targeted_vs_control':{
             kind:contrast('targeted','control',passed_pairs,kind) for kind in ('semantic','short','pair')}},
    }

    labels_path=HERE/'audit/semantic_review_blind.jsonl'
    rows=read_rows(labels_path)
    keys={r['review_id']:r for r in read_rows(HERE/'review-key-42.jsonl')}
    drafts={r['review_id']:r for r in read_rows(HERE/'review-42.jsonl')}
    assert len(keys)==len(rows)==len(drafts)==192
    assert {r['review_id'] for r in rows}==set(keys)==set(drafts)
    reviewed=defaultdict(list)
    for row in rows:
        rid=row['review_id']
        source=drafts[rid]
        key=keys[rid]
        assert row['id']==source['id'] and source['answer']==identities[row['id']]['answer']
        auto=predictions[key['model']][row['id']]
        assert key['parser_correct']==auto['semantic']
        reviewed[key['model']].append({**row,'parser_semantic':auto['semantic']})
    acceptance['blind_review']={'labels_sha256':sha(labels_path),'questions':48,'answers':192,
       'method':'Single Codex review with no model key before completing labels. Full original-image answers assessed against photographs and frozen annotations. Confirmed visible hallucinations and plain contradiction tags are conservative.',
       'by_model':{m:{'n':len(v),'scored_on_original_reference':{k:sum(r[k] for r in v) for k in METRICS},
                    'clear_labels_n':sum(not r['label_ambiguous'] for r in v),
                    'semantic_on_clear_labels':sum(r['semantic'] and not r['label_ambiguous'] for r in v),
                    'agreement':sum(r['semantic']==r['parser_semantic'] for r in v),
                    'parser_false_positive':sum(not r['semantic'] and r['parser_semantic'] for r in v),
                    'parser_false_negative':sum(r['semantic'] and not r['parser_semantic'] for r in v)} for m,v in reviewed.items()}}
    dump(HERE/'acceptance_summary.json',acceptance)
    print(acceptance)

if __name__=='__main__':
    main()
