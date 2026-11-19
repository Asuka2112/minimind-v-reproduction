"""Compute conservative semantic scores and fixed paired experimental gates."""
from collections import Counter, defaultdict
import argparse
from pathlib import Path
import sys
import re
import unicodedata

import numpy as np
from common import HERE, PRIOR, SEED, dump, dump_rows, load, read_rows, sha
sys.path.insert(1,str(PRIOR if PRIOR.exists() else HERE.parent / "targeted_sft"))
from build_data import ALIASES, NAMES

COLORS=['黑色','白色','红色','橙色','黄色','绿色','蓝色','紫色','灰色','粉色','棕色']
NUMBERS={'一':'1','二':'2','两':'2','三':'3','四':'4','五':'5','六':'6','七':'7','八':'8','九':'9','零':'0'}
OBJECT={alias:name for name in NAMES.values() for alias in [name,*ALIASES.get(name,())]}
COLOR={alias:name for name in COLORS for alias in (name,name[:-1])}
POSITION={alias:k for k in ('左边','右边') for alias in (k,*ALIASES[k])}
MEASURE=r'[个只位辆架头把件条本枚张盏艘匹套台双对支]'
NUM=r'[0-9一二两三四五六七八九零]'

def normalize(text):
    text=unicodedata.normalize('NFKC',text).strip()
    text=re.sub(r'^<think>.*?</think>\s*','',text,flags=re.S)
    text=re.sub(r'^\s*(?:\*{1,2}|[「“])*(?:答(?:案)?\s*[:：]|答案是)\s*','',text,count=1)
    return re.sub(r'\s+','',text.strip('。.!！?？；;:：,，"“”\'「」*')).casefold()

def dictionary_prefix(text,mapping):
    for name in sorted(mapping,key=lambda k:(-len(k),k)):
        if not text.startswith(name):
            continue
        trail=text[len(name):]
        if not trail or trail[0] in ',，。.!！?？;；:：、的在是位和及与':
            return mapping[name]
    return None

def relevant(text,q):
    task=q['task']
    start=normalize(text)
    if not start:
        return None
    if start in q['aliases']:
        return q['answer']
    if task=='count':
        concept=q['question'].split('几个',1)[-1].strip('？?')
        specific=[concept]+ALIASES.get(concept,[])
        for val in [start]+re.split(r'[。！？!?\n]',''.join(text.split()))[:2]:
            m=re.match(rf'^(?:一共有|总共|数量是|数量为|是|有)?({NUM})({MEASURE})?$',val)
            if m:
                return NUMBERS.get(m[1],m[1])
            for target in sorted(specific,key=len,reverse=True):
                m=re.search(rf'(?:有|共|见到|看到|数量为|数量是|^)(?:总共)?({NUM})(?:{MEASURE})?{re.escape(target)}',val)
                if m:
                    return NUMBERS.get(m[1],m[1])
        return None
    mapping={'object':OBJECT,'color':COLOR,'position':POSITION}[task]
    direct=dictionary_prefix(re.sub(r'^是(?:一'+MEASURE+')?','',start),mapping)
    if direct:
        return direct
    if task=='object':
        clauses=re.split('[。！？!?\n]',start)[:2]
        for clause in clauses:
            m=re.match(rf'^.{{0,20}}(?:蓝框|主要物体|主体).{{0,18}}?(?:是|为|看到|标出|框出)(?:一{MEASURE})?(.+)',clause)
            if m:
                found=dictionary_prefix(m[1],mapping)
                if found:
                    return found
        return None
    if task=='color':
        clauses=re.split('[。！？!?\n]',start)[:2]
        for clause in clauses:
            match=re.search(r'(?:主要颜色|颜色|主色)(?:主要)?(?:是|为|呈|有)(.{1,25})',clause)
            if match:
                found=dictionary_prefix(match[1],mapping)
                if found:
                    return found
        return None
    match=re.match(r'^.{0,12}蓝框.{0,21}(?:在|位于|处于)(?:画面)?(?:的)?(?:红框.{0,10}?的)?([左右](?:侧|边|方)?)',start)
    return mapping.get(match[1]) if match else None

def prior_evaluate_answer(raw,q):
    text=normalize(raw)
    # Contradictions counted when the answer itself gives incompatible alternatives.
    core=re.split(r'[。!！;；\n]',text)[0]
    contradiction=bool(re.search(r'(?:左边|左侧).{0,6}(?:或|或者|也可能).{0,4}(?:右边|右侧)|(?:右边|右侧).{0,6}(?:或|或者|也可能).{0,4}(?:左边|左侧)',core))
    prediction=relevant(raw,q) if not contradiction else None
    clauses=[s for s in re.split('[，,。！？!?；;]',text) if len(s)>=8]
    repetition=len(clauses)!=len(set(clauses)) or bool(re.search(r'(.{12,35})\1',text))
    # Only checks unsupported extra distinct classes, not all visual hallucinations.
    task=q['task']
    allowed=set(q['aliases'])
    extra=raw.count('。')>=2
    recognized_wrong=prediction is not None and prediction!=q['answer']
    hallucination_proxy=bool(extra and (recognized_wrong or contradiction))
    if task=='count':
        short_shape=bool(re.fullmatch(rf'{NUM}({MEASURE})?',text)) or (prediction is not None and not re.search(r'[，,;；。!?！？]',text))
    else:
        short_shape=dictionary_prefix(text,{'object':OBJECT,'color':COLOR,'position':POSITION}[task]) is not None or text in allowed
    short=bool(text and len(text)<=12 and short_shape)
    return {'semantic':prediction==q['answer'],'em':text in {normalize(x) for x in allowed},'short':short,
            'value':prediction,'repetition':repetition,'contradiction':contradiction,
            'hallucination_proxy':hallucination_proxy}

def ci(values):
    a=np.array(values,dtype=float)
    rng=np.random.default_rng(SEED)
    estimates=a[rng.integers(0,len(a),(2000,len(a)))].mean(1)*100
    return {'delta_pp':float(a.mean()*100),'ci95_pp':np.percentile(estimates,[2.5,97.5]).tolist()}

# Same scorer for the reused natural-photo test. These extensions apply only to the fresh challenge.
NEW = {
    '0':('零','没有','无','零个'), '5+':('至少5个','至少五个','5个或更多','五个或更多'),
    '上方':('上','上面','上边'), '下方':('下','下面','下边'),
    '前面':('前方','前','在前'), '后面':('后方','后','在后'),
    '更多':('多','较多'), '更少':('少','较少'), '一样多':('一样','相同','一样数量'),
}
RELATIONS={'horizontal':('左边','右边'),'vertical':('上方','下方'),'depth':('前面','后面'),
           'comparison':('更多','更少','一样多')}

def challenge_value(raw,q):
    task=q['subtask']
    text=normalize(raw)
    if task=='number':
        options=('0','1','2','3','4','5+')
        mapping={a:k for k in options for a in (k,*NEW.get(k,ALIASES.get(k,())))}
        if text in mapping:
            return mapping[text],False
        match=re.fullmatch(r'(?:图中|图里|一共|总共|答案是|有|是)*(?:至少)?([0-9]+|[零一二两三四五六七八九十])(?:[+个只辆条台])?',text)
        if match:
            n=int(NUMBERS.get(match[1],10 if match[1]=='十' else match[1]))
            return ('5+' if n>=5 else str(n)),False
        return None,False
    options=RELATIONS[task]
    mapping={a:k for k in options for a in (k,*NEW.get(k,ALIASES.get(k,())))}
    if text in mapping:
        return mapping[text],False
    alternatives='|'.join(re.escape(a) for a in sorted(mapping,key=lambda k:-len(k)) if len(a)>1)
    if task=='comparison':
        candidates=re.findall(r'(?:数量|比|相比|答案是|明显).{0,18}?('+alternatives+')',text)
    else:
        candidates=re.findall(r'蓝(?:色卡片|框物体|框).{0,7}?(?:位于|在|处于)红(?:色卡片|框物体|框)(?:的)?('+alternatives+')',text)
    keys={mapping[x] for x in candidates}
    if len(keys)>1:
        return None,True
    return (next(iter(keys)) if len(keys)==1 else dictionary_prefix(text,mapping)),False

def evaluate_answer(raw,q):
    if not q['id'].startswith('challenge-'):
        return prior_evaluate_answer(raw,q)
    text=normalize(raw)
    value,contradiction=challenge_value(raw,q)
    em=text in {normalize(x) for x in q['aliases']}
    clauses=[s for s in re.split('[，,。！？!?；;]',text) if len(s)>=8]
    repetition=len(clauses)!=len(set(clauses)) or bool(re.search(r'(.{12,35})\1',text))
    return {'semantic':value==q['answer'],'em':em,
            'short':bool(text and len(text)<=12 and value is not None and not contradiction
                         and not re.search(r'[，,;；。!?！？]',text)),
            'value':value,'repetition':repetition,'contradiction':contradiction,
            'hallucination_proxy':raw.count('。')>=2 and (contradiction or value is not None and value!=q['answer'])}

def summarize(split,seed,models):
    protocol=load(HERE/'protocol.json')
    protocol_sha=sha(HERE/'protocol.json')
    data=read_rows(HERE/f'data/{split}.jsonl')
    by_id={r['id']:r for r in data}
    pairs=defaultdict(list)
    for r in data:
        pairs[r['pair']].append(r['id'])
    assert all(len(v)==2 for v in pairs.values())
    all_items={}
    score={}
    text={}
    answer_mix={}
    for m in models:
        sid=42 if m=='base' else seed
        name=f'{split}-{m}-{sid}'
        path=HERE/f'predictions/{name}.jsonl'
        receipt=load(HERE/f'receipts/{name}.json')
        assert receipt['prediction_sha256']==sha(path) and receipt['protocol_sha256']==protocol_sha
        entries=read_rows(path)
        assert len(entries)==len(data)*2*3
        for x in entries:
            assert x['protocol_sha256']==protocol_sha and x['model']==m
            original=by_id[x['id']]
            shown=by_id[original['swapped_id']] if x['condition']=='swapped' else original
            assert x['image_sha256']==(None if x['condition']=='none' else shown['image_sha256'])
            k=(m,x['cap'],x['condition'],x['id'])
            assert k not in all_items
            all_items[k]={**x,**evaluate_answer(x['prediction'],original),
                          'shown_semantic':evaluate_answer(x['prediction'],shown)['semantic']}
        model={}
        tasks=sorted({r['task'] for r in data})
        subtasks=sorted({r['subtask'] for r in data if 'subtask' in r})
        for cap in protocol['eval']['caps']:
            conditions={}
            for condition in protocol['eval']['conditions']:
                aa=[all_items[m,cap,condition,r['id']] for r in data]
                keys=('semantic','shown_semantic','em','short','repetition','contradiction','hallucination_proxy','hit_length_cap')
                conditions[condition]={k:sum(p[k] for p in aa)/len(aa) for k in keys}
                for field,keys_ in [('task',tasks),('subtask',subtasks)]:
                    conditions[condition]['semantic_by_'+field]={
                        t:float(np.mean([p['semantic'] for r,p in zip(data,aa) if r.get(field)==t])) for t in keys_}
            both=[all(all_items[m,cap,'original',i]['semantic'] for i in ids) for ids in pairs.values()]
            original=[all_items[m,cap,'original',r['id']]['semantic'] for r in data]
            image_gaps={}
            for condition in ('swapped','none'):
                diffs=[np.mean([int(all_items[m,cap,'original',i]['semantic'])-
                                int(all_items[m,cap,condition,i]['semantic']) for i in ids])
                       for ids in pairs.values()]
                image_gaps[condition]=ci(diffs)
            assert len(original)==2*len(both)
            model[str(cap)]={'conditions':conditions,'pair_both_correct':float(np.mean(both)),
                             'pair_count':len(both),'original_minus_condition':image_gaps}
        score[m]=model
        if split=='test':
            txt=HERE/f'predictions/text-{m}-{sid}.jsonl'
            assert sha(txt)==receipt['text_sha256']
            rows=read_rows(txt)
            assert len({x['id'] for x in rows})==len(rows)==100
            text[m]={'n':100,'accuracy':sum(x['correct'] for x in rows)/100,
                     'gold_nll_per_token':float(np.mean([x['gold_nll_per_token'] for x in rows]))}
    report={'protocol_sha256':protocol_sha,'split':split,'seed':seed,'n':len(data),'rows':len(all_items),'models':score,'text':text}

    if {'control','targeted'}<=set(models):
        choice=load(HERE/'decode_selection.json')
        cap=choice['selected_cap']
        def contrast(a,b,metric,sub=None):
            tt=[ids for ids in pairs.values() if sub is None or sub in (by_id[ids[0]]['task'],by_id[ids[0]].get('subtask'))]
            assert tt
            if metric=='pair':
                differences=[int(all(all_items[a,cap,'original',i]['semantic'] for i in ids))-
                             int(all(all_items[b,cap,'original',i]['semantic'] for i in ids)) for ids in tt]
            else:
                differences=[np.mean([int(all_items[a,cap,'original',i][metric])-
                                      int(all_items[b,cap,'original',i][metric]) for i in ids]) for ids in tt]
            return {**ci(differences),'pairs':len(tt),'questions':2*len(tt)}
        comp={}
        for a,b in [('targeted','control'),('targeted','base'),('control','base')]:
            comp[f'{a}_vs_{b}']={key:contrast(a,b,key) for key in ('semantic','pair','short')}
            comp[f'{a}_vs_{b}']['by_task']={k:contrast(a,b,'semantic',k) for k in sorted({r['task'] for r in data})}
            comp[f'{a}_vs_{b}']['by_subtask']={k:contrast(a,b,'semantic',k) for k in sorted({r['subtask'] for r in data if 'subtask' in r})}
        report.update(selected_cap=cap,comparisons=comp)
        for m in models:
            answer_mix[m]={k:dict(Counter(all_items[m,cap,'original',r['id']]['value'] or '(unrecognized)' for r in data if r['task']==k))
                           for k in sorted({r['task'] for r in data})}
        report['normalized_response_counts']=answer_mix
        if split=='test':
            ab=comp['targeted_vs_control']
            min_text=min(text['targeted']['accuracy']-text[x]['accuracy'] for x in ('control','base'))*100
            limits=protocol['gates']
            gates={
                'semantic':ab['semantic']['delta_pp']>=limits['semantic_min_pp']-1e-9,
                'paired_semantic_ci':ab['semantic']['ci95_pp'][0]>0,
                'count_and_position':all(ab['by_task'][t]['delta_pp']>0 for t in ('count','position')),
                'both_images_correct':ab['pair']['delta_pp']>0,
                'text':min_text>=limits['text_min_pp']-1e-9,
            }
            report.update(gates=gates,all_gates=all(gates.values()),text_min_delta_pp=min_text)
        write_blind(split,seed,models,data,by_id,pairs,all_items,cap)
    dump(HERE/f'{split}-results-{seed}-{len(models)}models.json',report)
    return report

def write_blind(split,seed,models,data,by_id,pairs,all_items,cap):
    import random
    rgen=random.Random(SEED+31)
    pick=[]
    for task in sorted({r['task'] for r in data}):
        keys=[k for k,ids in pairs.items() if by_id[ids[0]]['task']==task]
        pick.extend(rgen.sample(sorted(keys),min(len(keys),4 if split=='test' else 3)))
    review=[]
    for p in pick:
        for ident in pairs[p]:
            for m in models:
                item=all_items[m,cap,'original',ident]
                r=by_id[ident]
                review.append(({'id':ident,'picture':r['image'],'question':r['question'],'answer':r['answer'],
                                'prediction':item['prediction']},{'model':m,'parser_correct':item['semantic']}))
    rgen.shuffle(review)
    dump_rows(HERE/f'review-{split}-{seed}.jsonl',[{'review_id':i+1,**r} for i,(r,_) in enumerate(review)])
    dump_rows(HERE/f'review-key-{split}-{seed}.jsonl',[{'review_id':i+1,**r} for i,(_,r) in enumerate(review)])

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('split',choices=['dev','test','challenge'])
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--models',nargs='+',default=['base','control','targeted'])
    args=parser.parse_args()
    result=summarize(args.split,args.seed,args.models)
    print(args.split,result['rows'],result.get('gates'),flush=True)
