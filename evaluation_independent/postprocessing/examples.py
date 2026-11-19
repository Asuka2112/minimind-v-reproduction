"""Transparent illustrative sample; selected by metadata hash, not answer correctness."""
import hashlib,json,shutil
from pathlib import Path
from PIL import Image
r=Path('/workspace/projects/minimind-v-independent-eval-20261103')
read=lambda p:[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
rows=read(r/'data/mmstar.jsonl')
models=['initial','pretrain','sft','official']
pred={m:{x['id']:x for x in read(r/f'predictions/{m}_correct.jsonl')} for m in models}
selected=[]
for c in sorted({x['category'] for x in rows}):
 selected.append(min((x for x in rows if x['category']==c),key=lambda x:hashlib.sha256(('examples20261103:'+x['id']).encode()).hexdigest()))
out=r/'examples';out.mkdir(exist_ok=True)
lines=['# 固定规则抽取的六个例子','','每个领域按 SHA256(`examples20261103:` + 样本 ID) 排序取首条，不根据对错筛选。这里只展示冻结的 greedy16 单字母答题协议输出；截断的描述文字不用于评价完整开放式问答质量。这不是新增人工标注集。','']
for row in selected:
 with Image.open(row['image_file']) as im:fmt=im.format.lower()
 ext='jpg' if fmt=='jpeg' else fmt
 dst=out/f'{row["id"]}.{ext}';shutil.copyfile(row['image_file'],dst)
 lines += [f'## {row["id"]} · {row["category"]}','',f'![{row["id"]}](examples/{dst.name})','',row['question'],'',f'数据集标准选项：**{row["answer"]}**','']
 for m in models:
  x=pred[m][row['id']]
  lines += [f'**{m}**（A–D 概率诊断：{x["restricted_choice"]}；达到长度上限：{x["hit_length_limit"]}）','', '```text',x['prediction'],'```','']
(r/'EXAMPLES.md').write_text('\n'.join(lines))
(r/'artifacts/example_selection.json').write_text(json.dumps({'selection':'one per category, minimum SHA256(examples20261103:+id); chosen after primary run, exploratory only','ids':[x['id'] for x in selected]},indent=2))
print('EXAMPLES_WRITTEN',[x['id'] for x in selected])
