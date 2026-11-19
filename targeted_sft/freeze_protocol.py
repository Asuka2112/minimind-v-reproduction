"""Freeze evaluation, data, and gate definitions before learning or test inference."""
from collections import Counter
from pathlib import Path
import random
from common import BASE, BASE_SHA, OFFICIAL, OFFICIAL_SHA, HERE, SOURCE, dump, dump_rows, load, now, read_rows, sha

TEXT_LISTS=[
 ('大雁', '大雁、石头、杯子', ['杯子','石头','大雁'], '哪一个是动物'),
 ('斑马','石头、斑马、火车',['火车','斑马','石头'],'哪一个是动物'),
 ('猫','猫、书、桌子',['书','桌子','猫'],'哪一个是动物'),
 ('狗','饼干、狗、键盘',['键盘','狗','饼干'],'哪一个是动物'),
 ('苹果','桌子、苹果、门',['门','苹果','桌子'],'哪一个是水果'),
 ('香蕉','香蕉、电梯、鞋',['鞋','电梯','香蕉'],'哪一个是水果'),
 ('橙子','石头、椅子、橙子',['椅子','石头','橙子'],'哪一个是水果'),
 ('梨','碗、梨、门',['门','梨','碗'],'哪一个是水果'),
 ('北京','北京、礼物、高兴',['高兴','北京','礼物'],'哪一个是城市'),
 ('杭州','电视、杭州、雨伞',['雨伞','杭州','电视'],'哪一个是城市'),
 ('七','六、七、八',['六','七','八'],'哪一个在六和八之间'),
 ('三','二、四、三',['二','四','三'],'哪一个在二和四之间'),
 ('白色','白色、椅子、跑步',['椅子','跑步','白色'],'哪一个表示颜色'),
 ('蓝色','帽子、蓝色、纸张',['帽子','蓝色','纸张'],'哪一个表示颜色'),
 ('游泳','树木、游泳、红色',['游泳','树木','红色'],'哪一个表示动作'),
 ('跑步','跑步、大象、苹果',['跑步','大象','苹果'],'哪一个表示动作'),
 ('春天','教室、春天、书架',['教室','春天','书架'],'哪一个是季节'),
 ('冬天','冬天、小猫、海洋',['海洋','小猫','冬天'],'哪一个是季节'),
 ('雨伞','雨伞、水杯、铅笔',['铅笔','水杯','雨伞'],'下雨时通常使用哪一个遮雨'),
 ('剪刀','米饭、剪刀、沙发',['剪刀','沙发','米饭'],'哪个物品通常用来剪纸'),
]

def make_text():
    rng=random.Random(3088)
    output=[]
    for k,(answer, context, opts, prompt) in enumerate(TEXT_LISTS):
        opts=opts.copy()
        rng.shuffle(opts)
        output.append({'id':f'text-{len(output):03}','kind':'category','question':f'在这些词里，{prompt}？{context}',
                       'answer':answer,'options':opts})
    for n,m in rng.sample([(a,b) for a in range(1,10) for b in range(1,10)],30):
        ans=str(n+m)
        opts=[ans,str(n+m+1),str(n+m-1)]
        rng.shuffle(opts)
        q=f'便签上写着：篮子里原有{n}个苹果，又放入{m}个。篮子里一共有几个苹果？'
        output.append({'id':f'text-{len(output):03}','kind':'count','question':q,'answer':ans,'options':opts})
    from itertools import permutations
    for j,values in enumerate(rng.sample(list(permutations('123456789',3)),50)):
        ans=values[j%3]
        names=['林兰','宋宁','高明']
        q='，'.join(f'{name}的编号是{v}' for name,v in zip(names,values))+f'。{names[j%3]}的编号是什么？'
        opts=list(values)
        rng.shuffle(opts)
        output.append({'id':f'text-{len(output):03}','kind':'retrieval','question':q,'answer':ans,'options':opts})
    assert len(output)==100
    return output

def main():
    out=HERE/'protocol.json'
    assert not out.exists()
    manifest=load(HERE/'data/manifest.json')
    groups={n:read_rows(HERE/f'data/{n}.jsonl') for n in ['dev','test','control','targeted']}
    assert {k:len(v) for k,v in groups.items()}=={'dev':80,'test':240,'control':2048,'targeted':2048}
    hashes={}
    for name,rows in groups.items():
        ids={r['coco_id'] for r in rows}
        hashes[name]=ids
        assert len(ids)==len(rows)
    assert hashes['control']==hashes['targeted']
    assert not (hashes['dev'] & hashes['test'] or hashes['dev'] & hashes['control'] or hashes['test'] & hashes['control'])
    regression=HERE/'data/text_regression.jsonl'
    dump_rows(regression,make_text())
    assert sha(BASE)==BASE_SHA and sha(OFFICIAL)==OFFICIAL_SHA
    source=SOURCE/'model'
    code={x.name:sha(x) for x in sorted(HERE.glob('*.py'))}
    config={
      'experiment':'评测','frozen_at':now(),
      'question':'Does changing the Chinese training task mixture improve semantic and short-answer outcomes while preserving visual grounding and text ability?',
      'inputs':{'base_native_sha256':BASE_SHA,'official_native_sha256':OFFICIAL_SHA},
      'upstream_file_sha256':{str(p.relative_to(source)):sha(p) for p in
        [source/'model_minimind.py',source/'model_vlm.py',source/'tokenizer_config.json',source/'tokenizer.json',
        *(source/'siglip2-base-p32-256-ve').glob('*')] if p.is_file()},
      'data_manifest_sha256':sha(HERE/'data/manifest.json'),'text_regression_sha256':sha(regression),
      'code_sha256':code,
      'training':{'examples_per_arm':2048,'epochs':2,'global_batch':16,'micro_batch':8,'updates_per_arm':256,
                  'update_cap':1000,'lr':5e-5,'random_seeds':[42,123],
                  'freeze':'projector + LM first and last layers; vision remains frozen','precision':'bf16 autocast; FP32 parameters',
                  'optimizer':'AdamW fused, cosine to 10% start lr, clip 1','number_of_gpus_per_arm':1,
                  'causal_intervention':'Same source photos and rendered JPEG bytes in the same sorted order. Mixed ordinary mostly-object QAs versus same photos with more count and position QAs. Answers concise in both. Test measures task-mixture intervention bundled across both tasks, no per-factor causal claim.',
                  'task_counts':{k:dict(Counter(r['task'] for r in groups[k])) for k in ('control','targeted')}},
      'smoke':{'rows':32,'steps':160,'gate':'Teacher-forced exact sequence >=24/32 and answer NLL at most half of before; safe plumbing smoke, not a held-out score.'},
      'eval':{'caps':[192,384],'default_cap':192,'decode':'FP32 deterministic greedy. Generate once up to 384 and evaluate prefix through 192 (same deterministic decode sequence). Main acceptance is fixed at 192.',
              'conditions':['original','swapped','none'],'swap':'Swap with different answer on the exact same question. Reindex already generated predictions for identical (question, image) inputs; hash validated.',
              'metrics':'semantic structured aliases + blind spot-check, normalized EM, short compliance independent of correctness (one direct candidate phrase of at most 12 characters), both images correct, original vs swapped original-target and no-image, repetitions, contradictions and potential hallucinations',
              'text':'100 fixed 3-way Chinese text probes, full candidate average log-likelihood (not greedy); separate report by category and gold NLL.',
              'semantic_scope':'Deterministic constrained-response parser for annotation-derived captions. Long/ambiguous responses may have unrecognized semantics. A blinded audit will record misparses separately.',
              'bootstrap':'2000 paired draws over complete two-image pairs, seed 1726.'},
      'gates':{'semantic_target_vs_control_at_least_pp':5.0,'short_target_vs_control_at_least_pp':10.0,
               'image_paired_semantic_gain_ci95_lower_gt_pp':0.0,
               'image_gain_definition':'targeted both-images-correct minus control both-images-correct, by image pair. Also report original minus swapped score at original targets.',
               'text_target_vs_control_min_pp':-2.0, 'text_target_vs_base_min_pp':-2.0,
               'second_seed':'Run seed 123 independent continuation iff seed 42 passes all gates.'}
    }
    dump(out,config)
    print('FROZEN',sha(out),flush=True)

if __name__=='__main__':
    main()
