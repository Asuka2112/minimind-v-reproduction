"""Compute conservative semantic scores and fixed paired experimental gates."""
from collections import Counter, defaultdict
import argparse
import random
import re
import unicodedata

import numpy as np
from build_data import ALIASES, NAMES
from common import HERE, SEED, dump, dump_rows, load, read_rows, sha

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

def evaluate_answer(raw,q):
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

def aggregate(split,seed):
    protocol=load(HERE/'protocol.json')
    protocol_sha=sha(HERE/'protocol.json')
    data=read_rows(HERE/f'data/{split}.jsonl')
    by_id={r['id']:r for r in data}
    pairs=defaultdict(list)
    for r in data:
        pairs[r['pair']].append(r['id'])
    models=['base','official'] if split=='dev' else ['base','official','control','targeted']
    all_items={}
    score={}
    text={}
    for m in models:
        # Baselines are evaluated once at the first seed.
        sid=42 if m in ('base','official') else seed
        path=HERE/f'predictions/{split}-{m}-{sid}.jsonl'
        receipt=load(HERE/f'receipts/{split}-{m}-{sid}.json')
        assert receipt['prediction_sha256']==sha(path) and receipt['protocol_sha256']==protocol_sha
        entries=read_rows(path)
        assert len(entries)==len(data)*2*3
        for x in entries:
            original=by_id[x['id']]
            shown=by_id[original['swapped_id']] if x['condition']=='swapped' else original
            assert x['image_sha256']==(None if x['condition']=='none' else shown['image_sha256'])
            k=(m,x['cap'],x['condition'],x['id'])
            assert k not in all_items
            all_items[k]={**x, **evaluate_answer(x['prediction'],original), 'shown_semantic':evaluate_answer(x['prediction'],shown)['semantic']}
        model={}
        for cap in protocol['eval']['caps']:
            conditions={}
            for condition in protocol['eval']['conditions']:
                aa=[all_items[m,cap,condition,r['id']] for r in data]
                keys=('semantic','shown_semantic','em','short','repetition','contradiction','hallucination_proxy','hit_length_cap')
                conditions[condition]={k:sum(p[k] for p in aa)/len(aa) for k in keys}
                conditions[condition]['semantic_by_task']={t:sum(p['semantic'] for r,p in zip(data,aa) if r['task']==t)/
                                                          sum(r['task']==t for r in data) for t in sorted({r['task'] for r in data})}
            both=[all(all_items[m,cap,'original',i]['semantic'] for i in ids) for ids in pairs.values()]
            model[str(cap)]={'conditions':conditions,'pair_both_correct':float(np.mean(both)),'pair_count':len(both)}
        score[m]=model
        if split=='test':
            txt=HERE/f'predictions/text-{m}-{sid}.jsonl'
            assert sha(txt)==receipt['text_sha256']
            tt=read_rows(txt)
            assert len(tt)==100
            text[m]={'n':len(tt),'accuracy':sum(r['correct'] for r in tt)/len(tt),
                     'gold_nll_per_token':float(np.mean([r['gold_nll_per_token'] for r in tt])),
                     'accuracy_by_task':{t:float(np.mean([r['correct'] for r in tt if r['kind']==t])) for t in sorted({r['kind'] for r in tt})}}
    report={'protocol_sha256':protocol_sha,'split':split,'seed':seed,'n':len(data),'rows':len(all_items),'models':score,'text':text}
    if split=='test':
        cap=protocol['eval']['default_cap']
        def ab(metric):
            return ci([np.mean([int(all_items['targeted',cap,'original',i][metric])-
                                int(all_items['control',cap,'original',i][metric]) for i in ids]) for ids in pairs.values()])
        both=ci([int(all(all_items['targeted',cap,'original',i]['semantic'] for i in ids))-
                 int(all(all_items['control',cap,'original',i]['semantic'] for i in ids)) for ids in pairs.values()])
        sem,short=ab('semantic'),ab('short')
        txt_a=100*(text['targeted']['accuracy']-text['control']['accuracy'])
        txt_base=100*(text['targeted']['accuracy']-text['base']['accuracy'])
        gates={'semantic':sem['delta_pp']>=5-1e-9,'short':short['delta_pp']>=10-1e-9,
               'visual_pair':both['ci95_pp'][0]>0,'text':min(txt_a,txt_base)>=-2-1e-9}
        report['comparisons']={'semantic':sem,'short':short,'both_images_correct':both,
                               'text_vs_control_pp':txt_a,'text_vs_base_pp':txt_base}
        report['gates']=gates
        report['all_gates']=all(gates.values())

        # Blind 20% of test pairs, stratified by task, from original predictions at 192.
        rng=random.Random(SEED)
        pick=[]
        for task in sorted({r['task'] for r in data}):
            available=[k for k,ids in pairs.items() if by_id[ids[0]]['task']==task]
            pick.extend(rng.sample(sorted(available),max(1,len(available)//5)))
        blind=[]
        for p in pick:
            for ident in pairs[p]:
                for m in models:
                    item=all_items[m,cap,'original',ident]
                    r=by_id[ident]
                    blind.append(({'id':ident,'picture':r['image'],'question':r['question'],'answer':r['answer'],
                                   'prediction':item['prediction']},{'model':m,'parser_value':item['value'],'parser_correct':item['semantic']}))
        rng.shuffle(blind)
        dump_rows(HERE/f'review-{seed}.jsonl',[{'review_id':i+1,**r} for i,(r,_) in enumerate(blind)])
        dump_rows(HERE/f'review-key-{seed}.jsonl',[{'review_id':i+1,**r} for i,(_,r) in enumerate(blind)])
    dump(HERE/f'{split}-results-{seed}.json',report)
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('split',choices=['dev','test'])
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    x=aggregate(args.split,args.seed)
    print(args.split,x['rows'],x.get('gates'),flush=True)
