"""Build paired Chinese VQA from pinned COCO photos and instance labels."""
import argparse
from collections import Counter, defaultdict
import io
import itertools
import json
from pathlib import Path
import random
import sqlite3

import numpy as np
from PIL import Image, ImageDraw
import pyarrow.parquet as pq
from common import COCO, HERE, OLD_INDEX, OLDER_EVAL, SEED, dump, dump_rows, image_fingerprint, load, sha

# Chinese names deliberately avoid ambiguous fine-grained labels.
NAMES = {
  1:'人',2:'自行车',3:'汽车',4:'摩托车',5:'飞机',6:'公交车',7:'火车',8:'卡车',9:'船',
  10:'红绿灯',11:'消防栓',13:'停车标志',15:'长椅',16:'鸟',17:'猫',18:'狗',19:'马',
  20:'羊',21:'牛',22:'大象',23:'熊',24:'斑马',25:'长颈鹿',27:'背包',28:'雨伞',31:'手提包',
  33:'行李箱',34:'飞盘',37:'球',38:'风筝',41:'滑板',42:'冲浪板',44:'瓶子',46:'酒杯',
  47:'杯子',48:'叉子',49:'刀',50:'勺子',51:'碗',52:'香蕉',53:'苹果',54:'三明治',
  55:'橙子',56:'西兰花',57:'胡萝卜',58:'热狗',59:'披萨',60:'甜甜圈',61:'蛋糕',
  62:'椅子',63:'沙发',64:'盆栽',65:'床',67:'餐桌',70:'马桶',72:'电视',73:'笔记本电脑',
  74:'鼠标',75:'遥控器',76:'键盘',77:'手机',78:'微波炉',79:'烤箱',81:'水槽',82:'冰箱',
  84:'书',85:'钟',86:'花瓶',87:'剪刀',88:'泰迪熊',89:'吹风机',90:'牙刷',
}
COUNTABLE = {1,2,3,4,5,6,7,9,15,16,17,18,19,20,21,22,24,25,27,28,33,34,37,38,42,
             44,46,47,51,52,53,55,60,62,64,70,72,73,84,85,86,88}
ALIASES = {
  '人':['人物','一个人','人类','男性','女性','男人','女人','男孩','女孩','小孩'],
  '汽车':['轿车','小汽车','车'], '公交车':['巴士','公共汽车','大巴'], '自行车':['单车'],
  '猫':['小猫','猫咪'], '狗':['小狗','狗狗'], '球':['球类','足球','篮球','网球','棒球'],
  '火车':['列车'], '船':['小船','轮船','帆船','船只'], '酒杯':['高脚杯','红酒杯'],
  '杯子':['水杯','茶杯','马克杯'], '笔记本电脑':['笔记本','手提电脑'],
  '左边':['左','左侧','左方'], '右边':['右','右侧','右方'],
  '1':['一','一个','一位','一只','一辆','一架','1个','1位','1只','1辆','1架'],
  '2':['二','两','两个','两位','两只','两辆','两架','2个','2位','2只','2辆','2架'],
  '3':['三','三个','三位','三只','三辆','三架','3个','3位','3只','3辆','3架'],
  '4':['四','四个','四位','四只','四辆','四架','4个','4位','4只','4辆','4架'],
  '5':['五','五个','五位','五只','五辆','五架','5个','5位','5只','5辆','5架'],
}
TASKS = ('object','color','count','position')
# Fixed before model predictions, after viewing 40 proposed test photos.
PREFLIGHT_REJECTS = {
    353970:'椅子框内猫更显眼，主体指代不够清楚',
    541634:'整张餐桌占框，食物和桌面颜色容易混淆',
    172877:'蓝框边缘处棕黄椅背，主色界限不清',
    144784:'人物只剩局部肢体，计数语义含糊',
    473974:'多只斑马严重遮挡，低分辨率下数量难以核验',
}

class HashIndex:
    def __init__(self):
        self.exacts = set()
        self.colors = defaultdict(set)
        self.parts = [defaultdict(set) for _ in range(4)]
    def add(self, rgb, dh):
        self.exacts.add(rgb)
        if dh not in self.colors:
            for i in range(4):
                self.parts[i][(dh>>(16*i))&65535].add(dh)
        self.colors[dh].add(rgb)
    def match(self, rgb, dh):
        if rgb in self.exacts:
            return 'exact'
        for i in range(4):
            for candidate in self.parts[i].get((dh>>(16*i))&65535, ()):
                if (candidate ^ dh).bit_count() <= 3:
                    return 'near'
        return None

def previous():
    index = HashIndex()
    audit = load(OLDER_EVAL/'artifacts/overlap_audit.json')
    assert len(audit['sources'])==2 and all(x['absent_from_old_full_sft_index']==0 for x in audit['sources'])
    connection = sqlite3.connect(f'file:{OLD_INDEX}?mode=ro',uri=True)
    for rgb, dh in connection.execute('select rgb_sha,dhash from image where valid=1'):
        index.add(rgb,int(dh,16))
    connection.close()
    return index

def dominant(image, annotation):
    polys=annotation['segmentation']
    if not isinstance(polys,list):
        return None
    mask=Image.new('L',image.size)
    pen=ImageDraw.Draw(mask)
    for p in polys:
        if len(p)>=6:
            pen.polygon(list(zip(p[::2],p[1::2])),fill=255)
    hsv=np.asarray(image.convert('HSV'))[np.asarray(mask)>0]
    if len(hsv)<1024:
        return None
    h,s,v=hsv.T
    chromatic=(s>=115)&(v>=75)
    bins={'红色':(h<=9)|(h>=246),'橙色':(h>=10)&(h<=25),'黄色':(h>=26)&(h<=47),
          '绿色':(h>=48)&(h<=105),'蓝色':(h>=128)&(h<=177),'紫色':(h>=182)&(h<=230)}
    rates={k:float((selector&chromatic).mean()) for k,selector in bins.items()}
    rates.update({'黑色':float((v<52).mean()), '白色':float(((v>205)&(s<44)).mean())})
    label=max(rates,key=rates.get)
    return label if rates[label] >= (.72 if label in ('黑色','白色') else .67) else None

def iou(a,b):
    x,y,w,h=a
    xx,yy,ww,hh=b
    intersection=max(0,min(x+w,xx+ww)-max(x,xx))*max(0,min(y+h,yy+hh)-max(y,yy))
    return intersection / (w*h+ww*hh-intersection)

def candidate_variants(image, items, image_id):
    width,height=image.size
    area=width*height
    rng=random.Random(SEED+image_id)
    valid=[a for a in items if not a['iscrowd'] and a['category_id'] in NAMES and
           a['area']>=.017*area and min(a['bbox'][2:])>=.055*min(image.size)]
    if not valid:
        return {}
    valid=sorted(valid,key=lambda a:-a['area'])[:6]
    obj=valid[0]
    result={}
    def item(task,question,answer,a,b=None):
        return {'task':task,'question':question,'answer':answer,'annotation_ids':[a['id']]+([b['id']] if b else []),
                'object_answer':NAMES[a['category_id']], 'boxes':[a['bbox']]+([b['bbox']] if b else [])}
    result['object']=item('object','蓝框标出的主要物体是什么？',NAMES[obj['category_id']],obj)
    for a in valid[:3]:
        if a['area'] < .035*area:
            continue
        color=dominant(image,a)
        if color:
            result['color']=item('color','蓝框中物体的主要颜色是什么？',color,a)
            break
    by_class=defaultdict(list)
    for a in items:
        by_class[a['category_id']].append(a)
    groups=[]
    for cls, instances in by_class.items():
        if cls not in COUNTABLE or not 1<=len(instances)<=5 or any(a['iscrowd'] for a in instances):
            continue
        if any(a['area']<.009*area or min(a['bbox'][2:])<.05*min(width,height) for a in instances):
            continue
        if any(iou(a['bbox'],b['bbox'])>.33 for a,b in itertools.combinations(instances,2)):
            continue
        groups.append((cls,instances))
    if groups:
        cls,instances=rng.choice(groups)
        result['count']=item('count',f'整张图里能看到几个{NAMES[cls]}？',str(len(instances)),max(instances,key=lambda a:a['area']))
    pairs=[]
    for a,b in itertools.combinations(valid,2):
        x,_,w,_=a['bbox']
        xx,_,ww,_=b['bbox']
        centers=(x+w/2)-(xx+ww/2)
        if abs(centers) >= .25*width and iou(a['bbox'],b['bbox']) == 0:
            pairs.append((a,b,centers))
    if pairs:
        a,b,d=rng.choice(pairs)
        if rng.random() < .5:
            a,b,d=b,a,-d
        result['position']=item('position','蓝框物体在红框物体的左边还是右边？','左边' if d<0 else '右边',a,b)
    return result

def render(image, boxes):
    width,height=image.size
    scale=min(480/width,480/height)
    canvas=Image.new('RGB',(480,480),(232,232,232))
    sw,sh=round(width*scale),round(height*scale)
    dx,dy=(480-sw)//2,(480-sh)//2
    canvas.paste(image.resize((sw,sh),Image.Resampling.LANCZOS),(dx,dy))
    draw=ImageDraw.Draw(canvas)
    for box,color in zip(boxes,['#165dff','#e5212e']):
        x,y,w,h=box
        draw.rectangle((dx+x*scale,dy+y*scale,dx+(x+w)*scale,dy+(y+h)*scale),outline=color,width=5)
    return canvas

def write_variant(root, raw, v, info, prefix):
    path=root/f'images/{prefix}.jpg'
    with Image.open(io.BytesIO(raw)) as im:
        render(im.convert('RGB'),v['boxes']).save(path,quality=91,subsampling=0)
    answer=v['answer']
    return {**{k:v[k] for k in ('task','question','answer','annotation_ids','boxes')},
            'image':str(path.relative_to(root)), 'image_sha256':sha(path), 'coco_id':info['id'],
            'original_rgb_sha256':info['rgb'], 'original_dhash':f"{info['dhash']:016x}",
            'license_id':info['license'], 'aliases':[answer]+ALIASES.get(answer,[answer.removesuffix('色')] if v['task']=='color' else [])}

def pair_pool(candidates, task, n, used, rng):
    # Prefer pairs of clearly different admissible answers to the exact same question.
    questions=defaultdict(list)
    for key, info in candidates.items():
        if task in info['variants']:
            questions[info['variants'][task]['question']].append(key)
    for keys in questions.values():
        rng.shuffle(keys)
    selected=[]
    category_cap=Counter()
    while len(selected)<n:
        best=None
        for q,keys in questions.items():
            available=[k for k in keys if k not in used]
            by_answer=defaultdict(list)
            for key in available:
                by_answer[candidates[key]['variants'][task]['answer']].append(key)
            for x,y in itertools.combinations(by_answer,2):
                priority=(category_cap[x]+category_cap[y],-min(len(by_answer[x]),len(by_answer[y])),rng.random())
                if best is None or priority<best[0]:
                    best=(priority,by_answer[x][0],by_answer[y][0],x,y)
        if best is None:
            raise RuntimeError(f'{task}: only {len(selected)} out of {n} disjoint pairs available')
        _,a,b,x,y=best
        if rng.randrange(2):
            a,b=b,a
        used.update((a,b))
        category_cap.update((x,y))
        selected.append((a,b))
    return selected

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=HERE/'data')
    parser.add_argument('--coco',type=Path,default=HERE/'coco')
    args=parser.parse_args()
    root=args.out
    assert not (root/'manifest.json').exists()
    (root/'images').mkdir(parents=True,exist_ok=True)
    for p, expected in COCO.items():
        assert sha(args.coco/p)==expected,p
    dataset=load(args.coco/'instances.json')
    metainfo={x['id']:x for x in dataset['images']}
    instances=defaultdict(list)
    for ann in dataset['annotations']:
        instances[ann['image_id']].append(ann)
    old=previous()
    local=HashIndex()
    by_id={}
    raw_data={}
    counter=Counter()
    variants=Counter()
    for p in ('val0.parquet','val1.parquet'):
        for batch in pq.ParquetFile(args.coco/p).iter_batches(batch_size=32,columns=['image_id','image']):
            for row in batch.to_pylist():
                key,raw=row['image_id'],row['image']['bytes']
                if key in PREFLIGHT_REJECTS:
                    counter['visual_preflight_rejected']+=1
                    continue
                meta=metainfo[key]
                with Image.open(io.BytesIO(raw)) as im:
                    image=im.convert('RGB')
                assert image.size==(meta['width'],meta['height'])
                rgb,dh=image_fingerprint(image)
                status=old.match(rgb,dh)
                if status:
                    counter[f'prior_{status}']+=1
                    continue
                status=local.match(rgb,dh)
                if status:
                    counter[f'within_{status}']+=1
                    continue
                record=candidate_variants(image,instances[key],key)
                if record:
                    local.add(rgb,dh)
                    raw_data[key]=raw
                    by_id[key]={'id':key,'rgb':rgb,'dhash':dh,'license':meta['license'],'variants':record}
                    variants.update(record.keys())
                else:
                    counter['not_eligible']+=1
        print('INGEST',p,len(by_id),dict(counter),dict(variants),flush=True)
    rng=random.Random(SEED)
    used=set()
    eval_data={}
    for split,n in [('test',30),('dev',10)]:
        rows=[]
        for task in ('count','color','position','object'):
            for i,pair in enumerate(pair_pool(by_id,task,n,used,rng)):
                pair_id=f'{split}-{task}-{i:02d}'
                duo=[]
                for j,key in enumerate(pair):
                    info=by_id[key]
                    row=write_variant(root,raw_data[key],info['variants'][task],info,f'{pair_id}-{j}')
                    row.update(id=f'{pair_id}-{j}',pair=pair_id)
                    duo.append(row)
                for a,b in ((duo[0],duo[1]),(duo[1],duo[0])):
                    assert a['question']==b['question'] and a['answer']!=b['answer']
                    a.update(swapped_id=b['id'])
                rows.extend(duo)
        dump_rows(root/f'{split}.jsonl',rows)
        eval_data[split]=rows
    # Match every source photo between A/B. A mixes common categories with a lower fraction of hard tasks.
    train_ids=set()
    train={'control':[],'targeted':[]}
    for task,n in [('color',288),('position',608),('count',608),('object',544)]:
        pool=[k for k,v in by_id.items() if k not in used and k not in train_ids and task in v['variants']]
        rng.shuffle(pool)
        # Reserve scarce color samples before the more common question types.
        selected=pool[:n]
        assert len(selected)==n,(task,n,len(selected))
        train_ids.update(selected)
        for j,key in enumerate(selected):
            info=by_id[key]
            variant=info['variants'][task]
            image_record=write_variant(root,raw_data[key],variant,info,f'train-{key}')
            first={**image_record,'id':f'train-{key}'}
            target=dict(first)
            control=dict(first)
            if task in ('count','position') and j>=208:
                control.update(task='object',question='蓝框标出的主要物体是什么？',answer=variant['object_answer'],
                               aliases=[variant['object_answer']]+ALIASES.get(variant['object_answer'],[]))
            train['control'].append(control)
            train['targeted'].append(target)
    for arm,rows in train.items():
        rng2=random.Random(SEED+1)
        rng2.shuffle(rows)
        dump_rows(root/f'{arm}.jsonl',rows)
    assert [r['image_sha256'] for r in train['control']]==[r['image_sha256'] for r in train['targeted']]
    assert not used & train_ids
    manifest={
        'source':'COCO val2017 photographs, repacked validation images and underlying official instance annotations',
        'mirrors':{'images':'detection-datasets/coco@cf0b22332314a937e9dc8a1957b21725430bb41d',
                   'annotations':'merve/coco@9e50abcdc1361852f34841af4939cbcd2d37c92f'},
        'source_sha256':COCO,'visual_preflight_rejects':PREFLIGHT_REJECTS,'rows_from_validation_shards':4952,'eligible_photos':len(by_id),'exclusions':dict(counter),
        'previous_overlap_scope':'Decoded RGB exact and dHash distance <=3 against existing complete-A/B training image index. The previous full-coverage audit proved both original training Parquets use encoded images present in that index. Resized/cropped and earlier foundation data contamination remain possible.',
        'in_round_overlap':'Distinct image IDs, exact decoded RGB hashes, and direct dHash <=3 filtered before splitting. Control/targeted train use exactly the same image sequence.',
        'splits':{k:dict(Counter(r['task'] for r in v)) for k,v in (eval_data|train).items()},
        'files':{p.name:sha(p) for p in root.glob('*.jsonl')},
        'rendering':'Real COCO photos letterboxed with reference boxes; custom annotation-derived benchmark, not standard COCO scores. Count targets all visible instances of named class. Color gold from dominant object mask; procedural candidates need spot review.',
    }
    dump(root/'manifest.json',manifest)
    print(manifest,flush=True)

if __name__=='__main__':
    main()
