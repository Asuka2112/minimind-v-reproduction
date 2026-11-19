"""Prepare matched A/B sets and a fresh count/spatial challenge from pinned COCO."""

from collections import Counter, defaultdict
from functools import lru_cache
import io
import random
import sys

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageOps

from common import COCO, HERE, PRIOR, dump, dump_rows, image_fingerprint, load, read_rows, sha
sys.path.insert(1, str(PRIOR))
from build_data import ALIASES, NAMES, HashIndex, candidate_variants, previous

BLUE=(30,105,246)
RED=(232,40,53)
CATEGORIES = {2,3,5,6,7,16,17,18,19,21,22,23,24,25,52,53,55,62,73,86}
COUNT_LABELS=('0','1','2','3','4','5+')
MORE_LABELS=('更多','更少','一样多')
SIMILAR=({2,3,5,6,7,9},{16,17,18,19,20,21,22,23,24,25},{44,47,51,52,53,55},{62,64,65,73,84,86})
ALIASES = dict(ALIASES)
ALIASES.update({'0':['零','没有','无','零个'],'5+':['至少5个','至少五个','5个或更多','五个或更多'],
                '上方':['上','上面','上边'],'下方':['下','下面','下边'],
                '前面':['前方','前','在前'],'后面':['后方','后','在后'],
                '更多':['多','较多'],'更少':['少','较少'],'一样多':['一样','相同','一样数量']})

def arow(task, question, answer, **info):
    return {'task':task,'question':question,'answer':answer,
            'aliases':[answer] + ALIASES.get(answer,[answer.removesuffix('色')] if task=='color' else []), **info}

def cutout(raw, ann):
    image=decode(raw)
    mask=Image.new('L',image.size,0)
    draw=ImageDraw.Draw(mask)
    for polygon in ann['segmentation']:
        if len(polygon)>=6:
            draw.polygon(list(zip(polygon[::2],polygon[1::2])),fill=255)
    bbox=mask.getbbox()
    if not bbox:
        raise ValueError(ann['id'])
    image=image.convert('RGBA')
    image.putalpha(mask)
    return image.crop(bbox)

def decode(raw):
    with Image.open(io.BytesIO(raw)) as p:
        return p.convert('RGB')

def save(image, ident, out):
    name=f'images/{ident}.jpg'
    image.convert('RGB').save(out/name,quality=94,subsampling=0)
    return {'id':ident,'image':name,'image_sha256':sha(out/name)}

def background(rng):
    # A neutral background with varied luminance. No numbers or answer-dependent colors.
    tone=rng.randint(218,250)
    tint=rng.choice([(0,0,0),(-5,-2,0),(0,-3,-7),(-1,0,-5)])
    return Image.new('RGB',(448,448),tuple(tone+t for t in tint))

def crop_panel(raw, variant):
    im=decode(raw)
    x,y,w,h=variant['boxes'][0]
    extra=.11
    crop=(max(0,int(x-w*extra)),max(0,int(y-h*extra)),
          min(im.width,int(np.ceil(x+w*(1+extra)))),min(im.height,int(np.ceil(y+h*(1+extra)))))
    bw,bh=crop[2]-crop[0],crop[3]-crop[1]
    assert bw>0 and bh>0
    im=im.crop(crop)
    sw,sh=ImageOps.contain(im,(412,412),Image.Resampling.LANCZOS).size
    dx,dy=(448-sw)//2,(448-sh)//2
    base=Image.new('RGB',(448,448),(234,234,234))
    base.paste(im.resize((sw,sh),Image.Resampling.LANCZOS),(dx,dy))
    sx,sy=sw/bw,sh/bh
    box=(dx+(x-crop[0])*sx,dy+(y-crop[1])*sy,dx+(x+w-crop[0])*sx,dy+(y+h-crop[1])*sy)
    ImageDraw.Draw(base).rectangle(box,outline=BLUE,width=6)
    return base

def shape(sprite, edge, rng):
    p=ImageOps.contain(sprite,(edge,edge),Image.Resampling.LANCZOS)
    if rng.randrange(2):
        p=ImageOps.mirror(p)
    return p

def pool_by_class(items, raw):
    rows=defaultdict(list)
    for ident,anns in items.items():
        if ident not in raw:
            continue
        for a in anns:
            if a['category_id'] in CATEGORIES and not a['iscrowd'] and isinstance(a['segmentation'],list):
                x,y,w,h=a['bbox']
                if w>=48 and h>=48 and a['area']>=2200 and a['area']/(w*h)>=.27:
                    rows[a['category_id']].append((ident,a))
    return {k:v for k,v in rows.items() if len({i for i,_ in v})>=2}

def donor_gallery(pools, raw_sets):
    out=HERE/'donor_qa'
    out.mkdir(exist_ok=True)
    data={}
    for split,pool in pools.items():
        raw=raw_sets[split]
        for k,options in sorted(pool.items()):
            items=[]
            used=set()
            # Prefer large instances that are away from the image border.
            def rank(option):
                i,a=option
                w,h=decode(raw[i]).size
                x,y,b,c=a['bbox']
                margin=min(x,y,w-x-b,h-y-c)/min(w,h)
                ratio=min(b,c)/max(b,c)
                return (margin>.025, min(.09,margin), ratio>.25, a['area'])
            for ident,ann in sorted(options,key=rank,reverse=True):
                if ident in used:
                    continue
                used.add(ident)
                items.append((ident,ann))
                if len(items)==12:
                    break
            gallery=Image.new('RGB',(1080,192),(238,238,238))
            pen=ImageDraw.Draw(gallery)
            name=f'{split}-{k}'
            for j,(ident,ann) in enumerate(items):
                col,row=j%12,j//12
                sprite=ImageOps.contain(cutout(raw[ident],ann),(86,150))
                gallery.paste(sprite,(col*90+(90-sprite.width)//2,row*192+max(3,(164-sprite.height)//2)),sprite)
                pen.text((col*90+6,row*192+169),str(j+1),fill='black')
            gallery.save(out/f'{name}.jpg',quality=88)
            data[name]=[(ident,a['id']) for ident,a in items]
    dump(out/'index.json',data)

PALETTE={'红色':(221,51,56),'橙色':(245,125,20),'黄色':(232,204,44),'绿色':(45,160,87),
         '蓝色':(43,137,218),'紫色':(152,68,195),'黑色':(31,35,40),'白色':(248,247,244)}

def color_panel(canvas,rng,index):
    names=list(PALETTE)
    label=names[index%len(names)]
    other=rng.choice([name for name in names if name!=label])
    centers=rng.sample(((105,109),(320,109),(105,328),(320,328)),2)
    pen=ImageDraw.Draw(canvas)
    boxes=[]
    for j,((cx,cy),name) in enumerate(zip(centers,(label,other))):
        cx,cy=cx+rng.randint(-22,22),cy+rng.randint(-22,22)
        w,h=rng.randint(42,74),rng.randint(40,67)
        box=(cx-w,cy-h,cx+w,cy+h)
        shape=rng.choice((pen.rounded_rectangle,pen.ellipse))
        shape(box,fill=PALETTE[name],outline=(51,57,69),width=3)
        if j==0:
            pen.rectangle((box[0]-8,box[1]-8,box[2]+8,box[3]+8),outline=BLUE,width=6)
        boxes.append(box)
    return arow('color','蓝框标出的图形主要是什么颜色？',label,subtask='marked_color',
                donors=[],color_box=boxes[0])

def choose_kind(cats,rng):
    if rng.random()<.36:
        groups=[sorted(g & cats.keys()) for g in SIMILAR if len(g & cats.keys())>=2]
        if groups:
            return rng.sample(rng.choice(groups),2)
    return rng.sample(sorted(cats),2)

def donor_pieces(raw,pool):
    @lru_cache(maxsize=2048)
    def get(ident, ann_id):
        anno=next(a for a in by_id[ident] if a['id']==ann_id)
        return cutout(raw[ident],anno)
    by_id=defaultdict(list)
    for xs in pool.values():
        for ident,ann in xs:
            by_id[ident].append(ann)
    def pick(key,rng,edge,used=()):
        available=[(ident,ann) for ident,ann in pool[key] if (ident,ann['id']) not in used]
        ident,x=rng.choice(available or pool[key])
        return shape(get(ident,x['id']),edge,rng),ident,x['id']
    return pick

def counter(canvas, pick, rng, cls, other, n, other_n, comparison):
    boxes=[]
    classes=[cls]*n+[other]*other_n
    rng.shuffle(classes)
    slots=list(range(9))
    rng.shuffle(slots)
    marked=None
    donors=[]
    occluded=[]
    for k,num in zip(classes,slots):
        row,col=divmod(num,3)
        cx,cy=76+col*148,76+row*148
        p,ident,aid=pick(k,rng,rng.randint(77,111),donors)
        x,y=cx-p.width//2+rng.randint(-4,4),cy-p.height//2+rng.randint(-4,4)
        box=(x,y,x+p.width,y+p.height)
        canvas.paste(p,(x,y),p)
        if k==cls and not occluded and rng.random()<.30:
            # A small foreground lip at the bottom/right extent. Require a real,
            # measured visible overlap between 2% and 18% of this instance mask.
            alpha=np.asarray(p.getchannel('A'))>128
            h,w=alpha.shape
            cut_y,cut_x=int(.65*h),int(.75*w)
            hidden=alpha[cut_y:,cut_x:].sum()/max(1,alpha.sum())
            if .02<=hidden<=.18:
                obstruction=(x+cut_x,y+cut_y,x+w+4,y+h+4)
                ImageDraw.Draw(canvas).rectangle(obstruction,fill=(133,140,146))
                occluded.append({'annotation':aid,'occluder':obstruction,'fraction':float(hidden)})
        donors.append((ident,aid))
        boxes.append({'cls':k,'box':box})
        if marked is None and k==other:
            marked=(x-5,y-5,x+p.width+5,y+p.height+5)
    assert len(classes)<=9 and marked is not None
    for i,a in enumerate(boxes):
        assert all(a['box'][2]<=b['box'][0] or b['box'][2]<=a['box'][0] or a['box'][3]<=b['box'][1] or b['box'][3]<=a['box'][1] for b in boxes[i+1:])
    ImageDraw.Draw(canvas).rectangle(marked,outline=BLUE,width=5)
    tag = 'comparison' if comparison else 'number'
    answer = (MORE_LABELS[0] if n>other_n else MORE_LABELS[1] if n<other_n else MORE_LABELS[2]) if comparison else ('5+' if n>=5 else str(n))
    if comparison:
        q=f'图里的{NAMES[cls]}比{NAMES[other]}更多、更少，还是一样多？'
    else:
        q=f'图里有几个{NAMES[cls]}？5个或更多回答5+。'
    b=arow('count',q,answer,subtask=tag,instance_counts={str(cls):n,str(other):other_n},
           instance_boxes=boxes,donors=donors,occlusions=occluded)
    a=arow('object','蓝框标出的主要物体是什么？',NAMES[other],subtask='marked_object',donors=donors)
    return a,b

def place(canvas, pick, rng, k, other, axis, answer):
    if axis=='depth':
        # Cards with opaque overlap supply visible front/back evidence. Target colors fixed.
        pen=ImageDraw.Draw(canvas)
        dx=rng.randint(-24,24)
        dy=rng.randint(-23,23)
        reverse=rng.randrange(2)==0
        left=(55+dx,90+dy,268+dx,300+dy)
        right=(168+dx,139+dy,387+dx,353+dy)
        blue,red=(right,left) if reverse else (left,right)
        objs=[(red,RED),(blue,BLUE)] if answer=='前面' else [(blue,BLUE),(red,RED)]
        for box,color in objs:
            pen.rounded_rectangle(box,radius=14,fill=color,outline=(24,32,41),width=5)
        assert min(blue[2],red[2])-max(blue[0],red[0])>60
        b=arow('position','两张卡片互相遮挡。蓝色卡片在红色卡片前面还是后面？',
               answer,subtask=axis,blue_box=blue,red_box=red,draw_order=['red','blue'] if answer=='前面' else ['blue','red'],donors=[])
        a=arow('color','红色卡片的主要颜色是什么？','红色',subtask='plain_color',donors=[])
        return a,b
    p,ida,aa=pick(k,rng,rng.randint(97,130))
    q,idb,ab=pick(other,rng,rng.randint(97,130))
    lo,hi=rng.randint(107,132),rng.randint(314,343)
    align=rng.randint(144,278)
    if axis=='horizontal':
        centers=((lo,align),(hi,align+rng.randint(-37,37)))
        question='蓝框物体在红框物体的左边还是右边？'
        first=answer=='左边'
    else:
        centers=((align,lo),(align+rng.randint(-37,37),hi))
        question='蓝框物体在红框物体的上方还是下方？'
        first=answer=='上方'
    if not first:
        centers=centers[::-1]
    boxes=[]
    for sprite,(cx,cy),color in zip((p,q),centers,(BLUE,RED)):
        x,y=cx-sprite.width//2,cy-sprite.height//2
        canvas.paste(sprite,(x,y),sprite)
        box=(x-6,y-6,x+sprite.width+6,y+sprite.height+6)
        ImageDraw.Draw(canvas).rectangle(box,outline=color,width=6)
        boxes.append(box)
    for box in boxes:
        assert min(box)>=0 and max(box)<448
    assert ((centers[0][0]<centers[1][0]) if axis=='horizontal' else (centers[0][1]<centers[1][1]))==first
    donors=[(ida,aa),(idb,ab)]
    b=arow('position',question,answer,subtask=axis,blue_box=boxes[0],red_box=boxes[1],donors=donors)
    a=arow('object','蓝框标出的主要物体是什么？',NAMES[k],subtask='marked_object',donors=donors)
    return a,b

def build(gallery_only=False):
    out=HERE/'data'
    if not gallery_only:
        assert not (out/'manifest.json').exists()
        (out/'images').mkdir(parents=True,exist_ok=True)
    coco=PRIOR/'coco'
    for name,want in COCO.items():
        assert sha(coco/name)==want,(name,sha(coco/name),want)
    ex_dev=read_rows(PRIOR/'data/dev.jsonl')
    ex_test=read_rows(PRIOR/'data/test.jsonl')
    ex_train=read_rows(PRIOR/'data/control.jsonl')
    all_ids={x['coco_id'] for x in ex_dev+ex_test+ex_train}
    old_train={x['coco_id'] for x in ex_train}
    assert len(old_train)==2048
    old_index=HashIndex()
    for x in ex_dev+ex_test+ex_train:
        old_index.add(x['original_rgb_sha256'],int(x['original_dhash'],16))
    full_training=previous()
    src=load(coco/'instances.json')
    items=defaultdict(list)
    for ann in src['annotations']:
        items[ann['image_id']].append(ann)
    raw_train={}
    raw_test={}
    fresh=HashIndex()
    exclusions=Counter()
    for file in ('val0.parquet','val1.parquet'):
        for batch in pq.ParquetFile(coco/file).iter_batches(batch_size=24,columns=['image_id','image']):
            for r in batch.to_pylist():
                i,raw=r['image_id'],r['image']['bytes']
                if i in old_train:
                    raw_train[i]=raw
                    continue
                if i in all_ids:
                    continue
                if not any(a['category_id'] in CATEGORIES and a['area']>=1800 for a in items[i]):
                    continue
                rgb,dh=image_fingerprint(decode(raw))
                if old_index.match(rgb,dh) or full_training.match(rgb,dh) or fresh.match(rgb,dh):
                    exclusions['prior_or_repeat']+=1
                    continue
                fresh.add(rgb,dh)
                raw_test[i]=raw
        print('ingest',file,len(raw_train),len(raw_test),flush=True)
    assert len(raw_train)==2048 and len(raw_test)>=180
    pool_train=pool_by_class(items,raw_train)
    pool_test=pool_by_class(items,raw_test)
    if gallery_only:
        donor_gallery({'train':pool_train,'challenge':pool_test},{'train':raw_train,'challenge':raw_test})
        return
    approved=load(HERE/'donor_qa/approved.json')
    catalog=load(HERE/'donor_qa/index.json')
    def curated(pool,split):
        output={}
        for cat,items_ in pool.items():
            key=f'{split}-{cat}'
            if key not in approved:
                continue
            allowed={tuple(catalog[key][i-1]) for i in approved[key]}
            selected=[(i,a) for i,a in items_ if (i,a['id']) in allowed]
            assert len(selected)==len(allowed) and len(selected)>=2,key
            output[cat]=selected
        return output
    pool_train=curated(pool_train,'train')
    pool_test=curated(pool_test,'challenge')
    assert pool_train.keys()==pool_test.keys() and len(pool_train)>=8
    choose_train=donor_pieces(raw_train,pool_train)
    choose_test=donor_pieces(raw_test,pool_test)
    variant={i:candidate_variants(decode(raw),items[i],i) for i,raw in raw_train.items()}
    rng=random.Random(48126)
    arm_a,arm_b=[],[]
    seen={x['image_sha256'] for x in ex_dev+ex_test}
    assert len(seen)==320
    def training(ident,image,a,b):
        common=save(image,ident,out)
        if common['image_sha256'] in seen:
            assert ident.startswith(('obj-','color-')),ident
            common=save(ImageOps.mirror(image),ident,out)
        assert common['image_sha256'] not in seen,ident
        seen.add(common['image_sha256'])
        arm_a.append({**common,**a})
        arm_b.append({**common,**b})

    eligible=[i for i in sorted(raw_train) if 'object' in variant[i]]
    rng.shuffle(eligible)
    for j,ident in enumerate(eligible[:800]):
        v=variant[ident]['object']
        x=arow('object',v['question'],v['answer'],subtask='annotated_object',donors=[(ident,v['annotation_ids'][0])])
        training(f'obj-{j:04d}',crop_panel(raw_train[ident],v),x,x)
    for j in range(640):
        can=background(rng)
        row=color_panel(can,rng,j)
        training(f'color-{j:04d}',can,row,row)

    # Conventional control answers a visible marked object on matched images.
    # It keeps a modest skewed subset of count/direction questions at their exact original labels.
    share_count={'0':6,'1':109,'2':48,'3':21,'4':8,'5+':4}
    n_kept=Counter()
    for j in range(960):
        cls,other=choose_kind(pool_train,rng)
        can=background(rng)
        if j<840:
            label=COUNT_LABELS[j%6]
            n=rng.randint(5,7) if label=='5+' else int(label)
            other_n=rng.randint(1,min(4,9-n))
            a,b=counter(can,choose_train,rng,cls,other,n,other_n,False)
            if n_kept[label]<share_count[label]:
                a=b
                n_kept[label]+=1
        else:
            relation=(j-840)%3
            n,other_n=((3,1),(1,3),(2,2))[relation]
            a,b=counter(can,choose_train,rng,cls,other,n,other_n,True)
            if j<870:
                a=b
        training(f'count-{j:04d}',can,a,b)
    spaces=[('horizontal','左边','右边',400,150),('vertical','上方','下方',280,10),('depth','前面','后面',120,0)]
    for axis,left,right,n,shared in spaces:
        for j in range(n):
            cls,other=choose_kind(pool_train,rng)
            can=background(rng)
            a,b=place(can,choose_train,rng,cls,other,axis,(left,right)[j%2])
            training(f'space-{axis}-{j:04d}',can,b if j<shared else a,b)

    assert len(arm_a)==len(arm_b)==3200
    order=list(range(len(arm_a)))
    random.Random(421).shuffle(order)
    arm_a=[arm_a[i] for i in order]
    arm_b=[arm_b[i] for i in order]
    assert [(r['id'],r['image_sha256']) for r in arm_a]==[(r['id'],r['image_sha256']) for r in arm_b]
    dump_rows(out/'control.jsonl',arm_a)
    dump_rows(out/'targeted.jsonl',arm_b)
    # Retain the first round's benchmarks using hard links on the same volume.
    for name,rs in [('dev',ex_dev),('test',ex_test)]:
        for row in rs:
            path=out/row['image']
            if not path.exists():
                path.hardlink_to(PRIOR/'data'/row['image'])
        dump_rows(out/f'{name}.jsonl',rs)
    text=read_rows(PRIOR/'data/text_regression.jsonl')
    dump_rows(out/'text_regression.jsonl',text)

    new=[]
    tr=random.Random(96014)
    def paired(id, render_a, render_b):
        duo=[]
        for k,(img,row) in enumerate((render_a,render_b)):
            rec=save(img,f'{id}-{k}',out)
            duo.append({**rec,**row,'pair':id,'swapped_id':f'{id}-{1-k}'})
        assert duo[0]['question']==duo[1]['question'] and duo[0]['answer']!=duo[1]['answer']
        new.extend(duo)
    for j in range(48):
        cls,other=choose_kind(pool_test,tr)
        scenarios=[]
        for label in (COUNT_LABELS[j%6], COUNT_LABELS[(j+1)%6]):
            n=tr.randint(5,7) if label=='5+' else int(label)
            can=background(tr)
            _,row=counter(can,choose_test,tr,cls,other,n,tr.randint(1,min(3,9-n)),False)
            scenarios.append((can,row))
        paired(f'challenge-number-{j:02d}',*scenarios)
    for j in range(12):
        cls,other=choose_kind(pool_test,tr)
        scenarios=[]
        for idx in (j%3,(j+1)%3):
            can=background(tr)
            x,z=((4,1),(1,4),(3,3))[idx]
            _,row=counter(can,choose_test,tr,cls,other,x,z,True)
            scenarios.append((can,row))
        paired(f'challenge-comparison-{j:02d}',*scenarios)
    for axis,left,right,*_ in spaces:
        for j in range(20):
            cls,other=choose_kind(pool_test,tr)
            scenarios=[]
            for label in (left,right):
                can=background(tr)
                _,row=place(can,choose_test,tr,cls,other,axis,label)
                scenarios.append((can,row))
            paired(f'challenge-{axis}-{j:02d}',*scenarios)
    assert len(new)==240
    all_train={i for row in arm_b for i,_ in row['donors']}
    all_challenge={i for row in new for i,_ in row['donors']}
    assert all_train <= old_train and not all_ids & all_challenge and not all_train & all_challenge
    assert len({r['image_sha256'] for r in arm_a+new+ex_dev+ex_test})==len(arm_a)+len(new)+len(ex_dev)+len(ex_test)
    dump_rows(out/'challenge.jsonl',new)
    counts=lambda rs: dict(Counter(r['task'] for r in rs))
    manifest={'files':{p.name:sha(p) for p in sorted(out.glob('*.jsonl'))},
              'source':'Pinned COCO photos and manually screened instance masks, generated color shapes and opaque-card occlusion scenes',
              'donor_selection_sha256':sha(HERE/'donor_qa/approved.json'),
              'donor_gallery_index_sha256':sha(HERE/'donor_qa/index.json'),
              'curated_class_count':len(pool_train),
              'instances_by_class':{name:{NAMES[k]:len(v) for k,v in pool.items()} for name,pool in
                                    (('train',pool_train),('challenge',pool_test))},
              'coco_sha256':COCO,'matching':'All 3200 encoded A/B images and order are identical. Prompts and answer labels differ in 1374 rows.',
              'old_test_state':'Frozen old comparison has previously been benchmarked. No per-item adaptation; evidence is corroborated on the separately locked challenge.',
              'challenge_scope':'Script-generated arrangements rather than unmodified natural photographs. Fresh source donors excluded by id/exact RGB/dHash<=3 against prior question pools and full original SFT visual index; foundation contamination unknown.',
              'train_donors':len(all_train),'challenge_donors':len(all_challenge),'old_train_pool':len(old_train),'fresh_pool':len(raw_test),
              'exclusions':dict(exclusions),'task_counts':{'control':counts(arm_a),'targeted':counts(arm_b),'challenge':counts(new)},
              'balanced_count':dict(Counter(x['answer'] for x in arm_b if x['task']=='count' and x['subtask']=='number')),
              'balanced_space':dict(Counter(x['subtask']+'/'+x['answer'] for x in arm_b if x['task']=='position')),
              'changed_questions':sum(a['question']!=b['question'] for a,b in zip(arm_a,arm_b))}
    manifest['mildly_occluded_count_scenes']=sum(bool(r.get('occlusions')) for r in arm_b)
    manifest['matching']=f"All 3200 encoded A/B images and order are identical. Prompts differ in {manifest['changed_questions']} rows."
    assert manifest['task_counts']['targeted']=={'object':800,'color':640,'count':960,'position':800}
    assert set(manifest['balanced_count'].values())=={140}
    dump(out/'manifest.json',manifest)
    print(manifest,flush=True)

if __name__=='__main__':
    build(gallery_only='--donor-gallery' in sys.argv[1:])
