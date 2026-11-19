"""Independently verify run receipts, sample pairing, and published result counts."""
from transformers import AutoTokenizer
from common import HERE, dump, load, now, read_rows, sha

def main():
    status=load(HERE/'status.json')
    root=HERE/'data'
    protocol=load(HERE/'protocol.json')
    manifest=load(root/'manifest.json')
    proto_sha=sha(HERE/'protocol.json')
    assert status['status']=='complete' and status['protocol_sha256']==proto_sha
    assert protocol['data_manifest_sha256']==sha(root/'manifest.json')
    for name,digest in protocol['code_sha256'].items():
        assert sha(HERE/name)==digest,name
    for name,digest in manifest['files'].items():
        assert sha(root/name)==digest,name
    dev=read_rows(root/'dev.jsonl')
    test=read_rows(root/'test.jsonl')
    control=read_rows(root/'control.jsonl')
    target=read_rows(root/'targeted.jsonl')
    assert list(map(len,[dev,test,control,target]))==[80,240,2048,2048]
    assert {r['coco_id'] for r in dev}.isdisjoint(r['coco_id'] for r in test+control)
    assert {r['coco_id'] for r in test}.isdisjoint(r['coco_id'] for r in control)
    assert [(r['id'],r['image_sha256']) for r in control]==[(r['id'],r['image_sha256']) for r in target]
    assert sum(a['question']!=b['question'] for a,b in zip(control,target))==800
    assert all(sha(root/r['image'])==r['image_sha256'] for r in dev+test)
    smoke=load(HERE/'receipts/smoke.json')
    assert smoke['pass_gate'] and smoke['rows']==32 and smoke['updates']==160
    trained=[]
    for seed in (42,123):
        result=HERE/f'test-results-{seed}.json'
        if not result.exists():
            continue
        arm_receipts=[load(HERE/f'receipts/{m}-{seed}.json') for m in ('control','targeted')]
        x,y=arm_receipts
        for k in ('seed','base_sha256','updates','rows','batch','micro_batch','learning_rate',
                  'freeze_policy','supervision','input_example_order_sha256','frozen_parameters_unchanged'):
            assert x[k]==y[k],k
        assert x['updates']==256 and x['rows']==2048
        result=load(result)
        assert result['protocol_sha256']==proto_sha and result['rows']==5760
        trained.append({'seed':seed,'gates':result['gates'],'all_gates':result['all_gates']})
    tokenizer_path=HERE.parent/'model'
    for name in ('tokenizer.json','tokenizer_config.json'):
        assert sha(tokenizer_path/name)==protocol['upstream_file_sha256'][name],name
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True)
    vocab_size=len(tokenizer)
    text_ids={row['id'] for row in read_rows(root/'text_regression.jsonl')}
    assert len(text_ids)==100
    assert sha(root/'text_regression.jsonl')==protocol['text_regression_sha256']
    records=0
    text_records=0
    for rfile in sorted((HERE/'receipts').glob('*.json')):
        rc=load(rfile)
        if not rc.get('prediction_sha256'):
            continue
        name=rfile.stem
        split=rc['split']
        data=dev if split=='dev' else test
        pred=HERE/f'predictions/{name}.jsonl'
        assert sha(pred)==rc['prediction_sha256']
        rs=read_rows(pred)
        index={(r['id'],r['condition'],r['cap']):r for r in rs}
        assert len(rs)==len(index)==len(data)*6==rc['rows']
        decoded=tokenizer.batch_decode([r['generated_ids'] for r in rs],skip_special_tokens=True)
        assert [r['prediction'] for r in rs]==decoded
        for truth in data:
            for c in ('original','swapped','none'):
                p,q=(index[truth['id'],c,cap] for cap in (192,384))
                assert p['generated_ids']==q['generated_ids'][:192]
                assert p['protocol_sha256']==q['protocol_sha256']==proto_sha
                for record in (p,q):
                    ids=record['generated_ids']
                    assert 1<=len(ids)<=record['cap'] and all(0<=v<vocab_size for v in ids)
                    assert record['hit_length_cap']==(len(ids)==record['cap'] and ids[-1]!=tokenizer.eos_token_id)
            assert (index[truth['id'],'swapped',384]['generated_ids']==
                    index[truth['swapped_id'],'original',384]['generated_ids'])
        if rc['text_sha256']:
            text_path=HERE/f"predictions/text-{rc['model']}-{rc['seed']}.jsonl"
            assert sha(text_path)==rc['text_sha256']
            text_rows=read_rows(text_path)
            assert len(text_rows)==len(text_ids) and {r['id'] for r in text_rows}==text_ids
            text_records+=len(text_rows)
        records+=len(rs)
    assert len(trained) in (1,2)
    assert records==6720+(2880 if len(trained)==2 else 0)
    x={'status':'verified','at':now(),'protocol_sha256':proto_sha,'vision_predictions_checked':records,
       'train_size_per_arm':2048,'switched_questions':800,'dev':len(dev),'test':len(test),
       'local_token_decodes':records,'text_regression_model_records':text_records,
       'smoke_before':smoke['before'],'smoke_after':smoke['after'],'seeds':trained,
       'scope':'Checks provenance and deterministic prediction relations. Semantic annotation is recorded separately.'}
    dump(HERE/'delivery_verification.json',x)
    print(x)

if __name__=='__main__':
    main()
