"""Pin the exact data, base weights, evaluation and success criteria."""
from common import BASE, BASE_SHA, HERE, PRIOR, SOURCE, dump, load, now, sha


def main():
    output = HERE / 'protocol.json'
    assert not output.exists()
    assert sha(BASE) == BASE_SHA
    manifest = HERE / 'data/manifest.json'
    auto = load(HERE/'quality.json')
    visual = load(HERE/'visual_quality.json')
    assert auto['pass'] and visual['pass']
    assert visual['sample_sha256'] == sha(HERE/'qa/sample.json')
    assert auto['manifest_sha256'] == visual['manifest_sha256'] == sha(manifest)
    upstream = SOURCE / 'model'
    upstream_paths = [upstream/name for name in ('model_minimind.py','model_vlm.py','tokenizer_config.json','tokenizer.json')]
    upstream_paths += sorted((upstream/'siglip2-base-p32-256-ve').glob('*'))
    assert len(upstream_paths) > 4
    dump(output, {
        'experiment':'评测', 'frozen_at':now(),
        'question':'Does balanced visual instruction data help Chinese counting and spatial understanding in a matched continuation?',
        'inputs':{'base_native_sha256':BASE_SHA, 'base_previous_arm':'targeted-42'},
        'code_sha256':{p.name:sha(p) for p in sorted(HERE.glob('*.py'))},
        'prior_source_sha256':{'build_data.py':sha(PRIOR/'build_data.py')},
        'upstream_file_sha256':{str(p.relative_to(upstream)):sha(p) for p in upstream_paths if p.is_file()},
        'data_manifest_sha256':sha(manifest),
        'quality_sha256':{name:sha(HERE/name) for name in ('quality.json','visual_quality.json')},
        'text_regression_sha256':sha(HERE/'data/text_regression.jsonl'),
        'training':{'examples_per_arm':3200, 'epochs':2, 'global_batch':16, 'micro_batch':8,
                    'updates_per_arm':400, 'update_cap':1000, 'lr':5e-5, 'random_seeds':[42,123],
                    'freeze':'Projector and first/last LM layers trainable; vision and all other LM parameters frozen.',
                    'precision':'BF16 autocast with FP32 master parameters',
                    'optimizer':'AdamW fused, cosine to 10% start LR, gradient clip 1',
                    'number_of_gpus_per_arm':1, 'task_counts':load(manifest)['task_counts'],
                    'comparison':'Shared rendered image bytes, sample order and hyperparameters. The intervention changes supervision task and answer distribution jointly.'},
        'smoke':{'rows':32, 'steps':160, 'lr':3e-4,
                 'gate':'Teacher-forced exact >=24/32, and supervised loss <=50% of initial loss.'},
        'eval':{'caps':[192,384], 'decode':'FP32 greedy with shared deterministic prefix',
                'selection':'Choose highest original-image base semantic accuracy on the frozen 80-item dev split. Break ties toward the shorter cap.',
                'conditions':['original','swapped','none'],
                'swap':'Identical question with its partner image whose gold differs; also score against the partner label.',
                'primary':'Existing frozen natural-photo test (240), previously measured on the starting checkpoint.',
                'secondary':'Fresh held-out transformed COCO challenge (240), with zero and 5+ counts, numerical comparison and three spatial axes.',
                'metrics':'Semantic, normalized EM, short compliance, both images correct, image-condition differences, repetition, contradiction, and heuristic hallucination.',
                'text':'100 three-way Chinese fixed text probes through candidate mean log likelihood.',
                'bootstrap':'2000 paired draws over complete pairs, seed 1726.'},
        'gates':{'semantic_min_pp':5., 'text_min_pp':-2.,
                 'paired_semantic_ci95_lower_exclusive_pp':0.,
                 'per_task_count_and_position_delta_strictly_positive':True,
                 'both_images_correct_delta_strictly_positive':True,
                 'second_seed':'Train seed 123 only if seed 42 passes all primary gates. Use the same selected decode cap.'}
    })
    print('FROZEN', sha(output), flush=True)


if __name__ == '__main__':
    main()
