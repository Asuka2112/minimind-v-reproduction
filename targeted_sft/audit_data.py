"""Sample and inspect 100 actual training rows without modifying the source."""
from collections import Counter, defaultdict
from pathlib import Path
import io
import json
import random
import re

from PIL import Image, ImageDraw, ImageOps
import pyarrow.parquet as pq
from common import CORPUS, CORPUS_SHA, HERE, SEED, dump, dump_rows, now, sha

def main():
    assert sha(CORPUS) == CORPUS_SHA
    p = pq.ParquetFile(CORPUS)
    sampled = sorted(random.Random(SEED).sample(range(p.metadata.num_rows), 100))
    by_group = defaultdict(list)
    base = 0
    for group in range(p.num_row_groups):
        count = p.metadata.row_group(group).num_rows
        by_group[group] = [(i, i-base) for i in sampled if base <= i < base+count]
        base += count
    rows, photos = [], []
    for group, indices in by_group.items():
        if not indices:
            continue
        table = p.read_row_group(group, columns=['conversations','image_bytes'])
        for index, offset in indices:
            row = {name: table[name][offset].as_py() for name in table.column_names}
            turns = json.loads(row['conversations'])
            q = ' '.join(t['content'] for t in turns if t['role']=='user').replace('<image>', '').strip()
            a = ' '.join(t['content'] for t in turns if t['role']=='assistant').strip()
            kind = ('count' if re.search(r'多少个|有几|几只|几位|How many',q,re.I) else
                    'position' if re.search(r'在.{0,15}(左边|右边|上方|下方)|左侧|右侧|located|position|relative to',q,re.I) else
                    'color' if re.search(r'颜色|colou?r',q,re.I) else
                    'description' if re.search(r'描述|说明.*图|caption|describe|what.*happening|image in detail',q,re.I) else 'other')
            info = {'row': index, 'question': q, 'answer': a, 'answer_chars': len(a),
                    'cjk_question': bool(re.search(r'[\u4e00-\u9fff]',q)), 'heuristic_topic': kind,
                    'placeholder_count': sum(t['content'].count('<image>') for t in turns if t['role']=='user')}
            try:
                with Image.open(io.BytesIO(row['image_bytes'])) as im:
                    image = im.convert('RGB')
                info['width'],info['height']=image.size
                info['blank'] = all(low==high for low,high in image.getextrema())
                photos.append((index, ImageOps.contain(image,(147,120))))
            except (TypeError, OSError, ValueError):
                info.update(decode_failure=True, blank=None)
                photos.append((index, None))
            rows.append(info)
    assert len(rows)==100
    out=HERE/'audit'
    out.mkdir(exist_ok=True)
    dump_rows(out/'sft_sample_100.jsonl',rows)
    for page in range(4):
        sheet=Image.new('RGB',(775,710),'white')
        draw=ImageDraw.Draw(sheet)
        for i,(index,img) in enumerate(photos[page*25:page*25+25]):
            x,y=(i%5)*155,(i//5)*142
            if img is not None:
                sheet.paste(img,(x+(155-img.width)//2,y))
            draw.text((x+8,y+121),f'{page*25+i+1:02} / {index}',fill='black')
        sheet.save(out/f'contact-{page+1}.jpg',quality=90)
    summary={'at':now(),'source_sha256':CORPUS_SHA,'sampling':'uniform unstratified sampling by source row, seed 1726',
             'source_rows':p.metadata.num_rows,'reviewed_rows':100,
             'topic_heuristics':dict(Counter(r['heuristic_topic'] for r in rows)),
             'cjk_questions':sum(r['cjk_question'] for r in rows),
             'answers_at_most_16_chars':sum(0<r['answer_chars']<=16 for r in rows),
             'answers_over_100_chars':sum(r['answer_chars']>100 for r in rows),
             'blank_images':sum(r['blank'] is True for r in rows),
             'decode_failures':sum(r.get('decode_failure',False) for r in rows),
             'placeholder_counts':dict(Counter(r['placeholder_count'] for r in rows)),
             'scope':'Topic counts are regex estimates; contact sheets and source text support separate alignment review.'}
    dump(out/'audit.json',summary)
    print(summary,flush=True)

if __name__=='__main__':
    main()
