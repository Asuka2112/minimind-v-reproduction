"""Audit the frozen inputs now and verify actual training and predictions at delivery."""
import argparse

from transformers import AutoTokenizer

from common import HERE, PRIOR, SOURCE, dump, load, now, read_rows, sha
from metrics import evaluate_answer


def verify_inputs():
    protocol_file = HERE/'protocol.json'
    protocol = load(protocol_file)
    manifest_file = HERE/'data/manifest.json'
    manifest = load(manifest_file)
    assert sha(manifest_file) == protocol['data_manifest_sha256']
    for name, expected in (protocol['code_sha256'] | protocol['quality_sha256']).items():
        assert sha(HERE/name) == expected, name
    source = SOURCE/'model' if SOURCE.is_dir() else HERE.parent/'model'
    prior = PRIOR if PRIOR.is_dir() else HERE.parent/'targeted_sft'
    assert sha(prior/'build_data.py') == protocol['prior_source_sha256']['build_data.py']
    for name, expected in protocol['upstream_file_sha256'].items():
        assert sha(source/name) == expected, name
    data = {}
    for filename, expected in manifest['files'].items():
        path = HERE/'data'/filename
        assert sha(path) == expected, filename
        data[path.stem] = read_rows(path)
    assert sha(HERE/'qa/sample.json') == load(HERE/'visual_quality.json')['sample_sha256']
    assert all(evaluate_answer(row['answer'], row)['semantic'] for row in data['challenge'])
    control, targeted = data['control'], data['targeted']
    assert len(control) == len(targeted) == protocol['training']['examples_per_arm']
    assert [(r['id'],r['image_sha256']) for r in control] == [(r['id'],r['image_sha256']) for r in targeted]
    assert sum(a['question'] != b['question'] for a,b in zip(control,targeted)) == manifest['changed_questions']
    train_ids = {ident for row in targeted for ident,_ in row['donors']}
    challenge_ids = {ident for row in data['challenge'] for ident,_ in row['donors']}
    assert train_ids.isdisjoint(challenge_ids)
    assert challenge_ids.isdisjoint(row['coco_id'] for row in data['dev']+data['test'])
    image_records = targeted+data['dev']+data['test']+data['challenge']
    assert len({row['image_sha256'] for row in image_records}) == len(image_records) == 3760
    if (HERE/'data/images').is_dir():
        for row in image_records:
            assert sha(HERE/'data'/row['image']) == row['image_sha256']
    return protocol, sha(protocol_file), manifest, source, data


def verify_predictions(name, rows, receipt, tokenizer, proto, conditions):
    prediction_file = HERE/f'predictions/{name}.jsonl'
    assert sha(prediction_file) == receipt['prediction_sha256']
    predictions = read_rows(prediction_file)
    assert len(predictions) == receipt['rows'] == len(rows)*len(conditions)*2
    entries = {(r['id'],r['condition'],r['cap']):r for r in predictions}
    assert len(entries) == len(predictions)
    assert tokenizer.batch_decode([r['generated_ids'] for r in predictions],skip_special_tokens=True) == [
        r['prediction'] for r in predictions
    ]
    images = {r['id']:r['image_sha256'] for r in rows}
    for row in rows:
        for condition in conditions:
            prefix, full = [entries[row['id'],condition,cap] for cap in (192,384)]
            assert prefix['generated_ids'] == full['generated_ids'][:192]
            shown = {'original':row['id'],'swapped':row['swapped_id'],'none':None}[condition]
            for item in (prefix, full):
                ids = item['generated_ids']
                assert item['protocol_sha256'] == proto and item['model'] == receipt['model']
                assert item['image_sha256'] == images.get(shown)
                assert 0 < len(ids) <= item['cap'] and all(0 <= token < len(tokenizer) for token in ids)
                assert item['hit_length_cap'] == (len(ids)==item['cap'] and ids[-1]!=tokenizer.eos_token_id)
            if condition=='swapped':
                assert full['generated_ids'] == entries[shown,'original',384]['generated_ids']
    return entries


def verify_results(protocol, proto, manifest, source, data):
    state = load(HERE/'status.json')
    assert state['status']=='complete' and state['protocol_sha256']==proto
    smoke = load(HERE/'receipts/smoke.json')
    assert smoke['protocol_sha256']==proto and smoke['updates']==protocol['smoke']['steps']
    assert smoke['pass_gate'] and smoke['after']['n']==32
    assert smoke['after']['teacher_forced_exact_examples']>=24 and smoke['after']['nll']<=.5*smoke['before']['nll']
    choice = load(HERE/'decode_selection.json')
    selected_cap = choice['selected_cap']
    assert choice['protocol_sha256']==proto and selected_cap in protocol['eval']['caps']
    tokenizer = AutoTokenizer.from_pretrained(source,local_files_only=True)
    seeds = (42,123) if state['first_seed_passed'] else (42,)
    tasks = {('dev','base',42)}
    hashes = {('base',42):protocol['inputs']['base_native_sha256']}
    training = protocol['training']
    for seed in seeds:
        receipts = []
        for arm in ('control','targeted'):
            receipt = load(HERE/f'receipts/{arm}-{seed}.json')
            assert receipt['protocol_sha256']==proto and receipt['seed']==seed and receipt['arm']==arm
            assert receipt['base_sha256']==hashes['base',42]
            assert receipt['data_sha256']==manifest['files'][arm+'.jsonl']
            assert receipt['frozen_parameters_unchanged'] and receipt['changed_trainable_tensors']
            for x,y in [('rows','examples_per_arm'),('updates','updates_per_arm'),('batch','global_batch'),
                        ('micro_batch','micro_batch'),('learning_rate','lr')]:
                assert receipt[x]==training[y], (x,y)
            assert [x['step'] for x in receipt['losses']]==list(range(1,receipt['updates']+1))
            receipts.append(receipt)
            file = HERE/f'weights/{arm}-{seed}.pth'
            if file.is_file():
                assert sha(file)==receipt['checkpoint_sha256']
            hashes[arm,seed] = receipt['checkpoint_sha256']
            tasks.update({('test',arm,seed), ('challenge',arm,seed)})
        assert receipts[0]['input_example_order_sha256']==receipts[1]['input_example_order_sha256']
    tasks.update({('test','base',42),('challenge','base',42)})
    total, text_total = 0,0
    actual = {}
    for split, model, seed in sorted(tasks):
        name = f'{split}-{model}-{seed}'
        receipt = load(HERE/f'receipts/{name}.json')
        assert receipt['protocol_sha256']==proto and receipt['checkpoint_sha256']==hashes[model,seed]
        assert (receipt['split'],receipt['model'],receipt['seed'])==(split,model,seed)
        rows = data[split]
        entries = verify_predictions(name, rows, receipt, tokenizer, proto, protocol['eval']['conditions'])
        total += len(entries)
        if split=='test':
            file = HERE/f'predictions/text-{model}-{seed}.jsonl'
            assert sha(file)==receipt['text_sha256']
            labels = {r['id']:r['answer'] for r in data['text_regression']}
            records = read_rows(file)
            assert len(records)==len(labels)==100 and {r['id'] for r in records}==labels.keys()
            for row in records:
                assert row['answer']==labels[row['id']]
                assert row['correct']==(row['prediction']==row['answer'])
            text_total += len(records)
        cap_scores = {}
        for cap in protocol['eval']['caps']:
            flags = {r['id']:evaluate_answer(entries[r['id'],'original',cap]['prediction'],r)['semantic'] for r in rows}
            cap_scores[cap] = {
                'semantic':sum(flags.values())/len(rows),
                'pair':sum(flags[r['id']] and flags[r['swapped_id']] for r in rows)/len(rows),
            }
        actual[split,model,seed] = cap_scores
    base_dev = {k:v['semantic'] for k,v in actual['dev','base',42].items()}
    assert base_dev=={int(k):v for k,v in choice['dev_semantic'].items()}
    assert selected_cap==max(base_dev,key=lambda cap:(base_dev[cap],-cap))
    for seed in seeds:
        for split in ('test','challenge'):
            result = load(HERE/f'{split}-results-{seed}-3models.json')
            assert result['protocol_sha256']==proto and result['selected_cap']==selected_cap
            for model in ('base','control','targeted'):
                checked = actual[split,model,42 if model=='base' else seed]
                for cap, values in checked.items():
                    scored = result['models'][model][str(cap)]
                    assert values['semantic']==scored['conditions']['original']['semantic']
                    assert values['pair']==scored['pair_both_correct']
            if split=='test':
                key = 'first_seed' if seed==42 else 'second_seed'
                assert state[key+'_gates']==result['gates'] and state[key+'_passed']==all(result['gates'].values())
    assert total==9120+(5760 if 123 in seeds else 0)
    assert text_total==300+(200 if 123 in seeds else 0)
    return {'formal_seeds':seeds,'selected_cap':selected_cap,
            'vision_predictions_verified':total,'text_predictions_verified':text_total,
            'weight_files_hashed':sum((HERE/f'weights/{m}-{s}.pth').is_file() for m,s in hashes if m!='base')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs-only', action='store_true')
    args = parser.parse_args()
    protocol, proto, manifest, source, data = verify_inputs()
    result = {'checked_at':now(),'protocol_sha256':proto,
              'status':'frozen_inputs_verified' if args.inputs_only else 'run_artifacts_verified',
              'split_sizes':{key:len(value) for key,value in data.items()}}
    if not args.inputs_only:
        result.update(verify_results(protocol,proto,manifest,source,data))
        dump(HERE/'delivery_verification.json',result)
    print(result)


if __name__=='__main__':
    main()
