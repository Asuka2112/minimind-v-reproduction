"""Freeze new Chinese questions with gold answers and counterfactual images."""
import argparse
import hashlib
import json
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps

HERE = Path(__file__).resolve().parent
REFERENCE = Path('/workspace/projects/minimind-v-independent-eval-20261103')
TRAIN = Path('/workspace/projects/minimind-v-official-20261102')
MODELS = {
    'official': (TRAIN / 'weights/official_sft_vlm_768.pth', '0d3fee0970c2174d22226de25d273e2f375bda8ed39b8f1673e1519c94264f9e'),
    'reproduced': (TRAIN / 'runs/sft/out/sft_vlm_768.pth', '37d2450149c1c9d6f72d03780cd71abb04174352ca02f01bfb4a8843cc0892c4'),
}
COLORS = {'红色': '#d52632', '蓝色': '#216acf', '绿色': '#219752', '黄色': '#e8b51b'}
SHAPES = ('圆形', '正方形', '三角形', '五角星')
PLACES = ('左边', '右边', '上方', '下方')
NATURAL = [
    ('物体', '画面里最显眼的动物是什么？', ('0224', '大象'), ('0025', '马')),
    ('物体', '图片里是什么房间？', ('0311', '厨房'), ('0078', '浴室')),
    ('物体', '图中主要人物正在使用什么乐器或设备？', ('0060', '麦克风'), ('0407', '架子鼓')),
    ('颜色', '图片中的天空主要是什么颜色？', ('0360', '蓝色'), ('0183', '灰色')),
    ('颜色', '图中主要人物的上衣主要是什么颜色？', ('0348', '深蓝色'), ('0226', '白色')),
    ('颜色', '图片近处的地面主要是什么颜色？', ('0037', '白色'), ('0003', '绿色')),
    ('计数', '图中有几个人坐着？', ('0290', '2'), ('0287', '1')),
    ('计数', '图中清楚可见的人有几位？', ('0309', '2'), ('0362', '1')),
    ('计数', '有几个足球接触草地？', ('0027', '2'), ('0136', '1')),
]
MIRRORS = [
    ('0429', '黑色笔记本电脑主要在画面的左边还是右边？', '左边', '右边'),
    ('0172', '列车主要在画面的左边还是右边？', '右边', '左边'),
    ('0324', '香蕉主要在画面的左边还是右边？', '右边', '左边'),
]
ALIASES = {
    '圆形': ['圆', '圆圈'], '正方形': ['方形', '方块'], '三角形': ['三角'],
    '五角星': ['星形', '星星'], '红色': ['红'], '蓝色': ['蓝'], '绿色': ['绿'],
    '黄色': ['黄'], '灰色': ['灰'], '白色': ['白'], '深蓝色': ['深蓝', '藏蓝色', '藏青色', '蓝色', '蓝'],
    '左边': ['左', '左侧', '左方'], '右边': ['右', '右侧', '右方'],
    '上方': ['上', '上边', '上面', '上侧'], '下方': ['下', '下边', '下面', '下侧'],
    '大象': ['象'], '马': ['白马', '一匹马'], '厨房': ['开放式厨房'],
    '浴室': ['洗手间', '卫生间', '浴室卫生间'], '麦克风': ['话筒', '麦'],
    '架子鼓': ['爵士鼓', '鼓', '鼓组', '打鼓'], '1': ['一', '一个', '一位', '1个', '1位', '一人', '1人'],
    '2': ['二', '两', '两个', '两位', '二个', '2个', '2位', '两人', '2人'],
    '3': ['三', '三个', '三位', '3个', '3位', '三人', '3人'],
    '4': ['四', '四个', '四位', '4个', '4位', '四人', '4人'],
}

def sha(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def write(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)

def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]

def shape(draw, name, x, y, r, color):
    if name == '圆形':
        draw.ellipse((x-r, y-r, x+r, y+r), fill=color)
    elif name == '正方形':
        draw.rectangle((x-r, y-r, x+r, y+r), fill=color)
    elif name == '三角形':
        draw.polygon(((x, y-r), (x-r, y+r), (x+r, y+r)), fill=color)
    else:
        from math import cos, sin, tau
        points = [(x + cos(tau*i/10-tau/4)*r*(1 if i%2 == 0 else .44),
                   y + sin(tau*i/10-tau/4)*r*(1 if i%2 == 0 else .44)) for i in range(10)]
        draw.polygon(points, fill=color)

def generate_synthetic(root, seed):
    rng = random.Random(seed)
    rows = []
    cases = [
        ('物体', '画面中间是什么形状？', SHAPES),
        ('颜色', '画面中的图形是什么颜色？', tuple(COLORS)),
        ('计数', '画面里有几个圆形？', ('1', '2', '3', '4')),
        ('位置', '蓝色圆形位于灰色正方形的哪个方向？', PLACES),
    ]
    for category_index, (category, question, answers) in enumerate(cases):
        answer_pairs = [(answers[i], answers[(i+step) % 4]) for step in (1, 2, 3, 1) for i in range(4)]
        rng.shuffle(answer_pairs)
        for i, answer_pair in enumerate(answer_pairs):
            pair = f's{category_index}-{i:02}'
            bg = rng.choice(['#f9f9f9', '#eee9df', '#e9eef1', '#faf0e5'])
            radius = rng.randint(36, 53)
            x, y = 160+rng.randint(-17, 17), 160+rng.randint(-17, 17)
            color = rng.choice(list(COLORS.values()))
            targets = [(xx+rng.randint(-12,12), yy+rng.randint(-12,12)) for yy in (77, 235) for xx in (77, 235)]
            rng.shuffle(targets)
            for j, answer in enumerate(answer_pair):
                im = Image.new('RGB', (320, 320), bg)
                draw = ImageDraw.Draw(im)
                if category == '物体':
                    shape(draw, answer, x, y, radius+16, color)
                elif category == '颜色':
                    shape(draw, SHAPES[i%4], x, y, radius+17, COLORS[answer])
                elif category == '计数':
                    for xx, yy in targets[:int(answer)]:
                        shape(draw, '圆形', xx, yy, radius, color)
                else:
                    direction = {'左边':(-93,0), '右边':(93,0), '上方':(0,-93), '下方':(0,93)}
                    dx, dy = direction[answer]
                    shape(draw, '正方形', x, y, 40, '#7e7e7e')
                    shape(draw, '圆形', x+dx, y+dy, 37, COLORS['蓝色'])
                rel = f'images/{pair}-{j}.png'
                im.save(root / rel)
                rows.append(make_row(pair, j, 'synthetic', category, question, answer, rel, 'programmatic'))
    return rows

def make_row(pair, j, source, category, question, answer, rel, provenance):
    return {'id': f'{pair}-{j}', 'pair': pair, 'source': source, 'category': category, 'question': question,
            'answer': answer, 'aliases': [answer] + ALIASES.get(answer, []), 'image': rel, 'provenance': provenance}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Verify synthetic rendering in a temporary folder')
    args = parser.parse_args()
    if args.dry_run:
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'images').mkdir()
            rows = generate_synthetic(root, 1726)
            assert len(rows) == 128 and len({sha(root / r['image']) for r in rows}) == 128
            print('128 synthetic images verified')
        return
    assert not (HERE / 'protocol.json').exists() and not (HERE / 'predictions').exists()
    data = HERE / 'data'
    (data / 'images').mkdir(parents=True, exist_ok=True)
    old = {r['id']: r for r in read_rows(REFERENCE / 'data/mmstar.jsonl')}
    overlap_path = REFERENCE / 'artifacts/overlap_audit.json'
    overlap = json.loads(overlap_path.read_text())
    rows = generate_synthetic(data, 1726)

    def get_photo(image_id, filename, mirror=False):
        key = f'mmstar-{image_id}'
        row = old[key]
        assert overlap['row_masks'][key]['conservative_free']
        raw = Path(row['image_file'])
        assert sha(raw) == row['image_file_sha256']
        rel = 'images/' + filename + '.png'
        with Image.open(raw) as image:
            rgb = image.convert('RGB')
            if mirror:
                rgb = ImageOps.mirror(rgb)
            rgb.save(data / rel)
        return rel, {'id': key, 'original_sha256': sha(raw), 'mirror': mirror}

    for i, (category, q, a, b) in enumerate(NATURAL):
        pair = f'n-{i:02}'
        for j, (image_id, answer) in enumerate((a,b)):
            rel, provenance = get_photo(image_id, f'{pair}-{j}')
            rows.append(make_row(pair, j, 'natural', category, q, answer, rel, provenance))
    for i, (image_id, q, a, b) in enumerate(MIRRORS, len(NATURAL)):
        pair = f'n-{i:02}'
        for j, answer in enumerate((a,b)):
            rel, provenance = get_photo(image_id, f'{pair}-{j}', mirror=j == 1)
            rows.append(make_row(pair, j, 'natural_mirror', '位置', q, answer, rel, provenance))
    assert len(rows) == 152 and len({r['image'] for r in rows}) == 152
    for i in range(0, len(rows), 2):
        a, b = rows[i:i+2]
        assert a['pair'] == b['pair'] and a['question'] == b['question']
        assert not set(a['aliases']) & set(b['aliases'])
        for row, other in ((a,b),(b,a)):
            row.update(image_sha256=sha(data / row['image']), swapped_image=other['image'],
                       swapped_answer=other['answer'], swapped_aliases=other['aliases'])
        assert a['image_sha256'] != b['image_sha256']
    dataset = data / 'questions.jsonl'
    dataset.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
    for path, expected in MODELS.values():
        assert sha(path) == expected
    source = TRAIN / 'src/model'
    protocol = {
        'name': '评测', 'frozen_at': datetime.now(timezone.utc).isoformat(),
        'rows': len(rows), 'pair_count': len(rows)//2,
        'groups': {'synthetic': 128, 'natural': 18, 'natural_mirror': 6},
        'natural_scope': 'Purposive retrospective subset of previously used external images, new Chinese questions and image-verified labels. Training overlap flags are from the prior complete image audit. Mirror examples are derived. Not a fresh external benchmark.',
        'synthetic_scope': 'New deterministic geometric scenes with known labels. 32 samples per task, answer labels balanced across the four values. Diagnostic of basic visual grounding, not realistic QA.',
        'source_commit': '041d528e037a8a373c3c8aa7278a3ba3bd26b6dc',
        'prior_overlap_audit_sha256': sha(overlap_path),
        'models': {k: {'checkpoint_sha256': digest} for k, (_, digest) in MODELS.items()},
        'model_source_sha256': {str(p.relative_to(source)): sha(p) for p in
                                [source/'model_minimind.py', source/'model_vlm.py', source/'tokenizer.json',
                                 source/'tokenizer_config.json', *(source/'siglip2-base-p32-256-ve').glob('*')] if p.is_file()},
        'dataset_sha256': sha(dataset), 'code_sha256': {p.name: sha(p) for p in HERE.glob('*.py')},
        'conditions': ['correct', 'swapped', 'none'],
        'prompts': {'baseline': '', 'short': '\n请用中文简短回答，只给出答案。'},
        'decode': {'greedy': True, 'max_new_tokens': 192, 'fp32': True, 'tf32': False, 'image_tokens': 64},
        'primary_score': 'Normalized accepted short answer. NFKC, remove outer punctuation, one answer prefix and leading 是/为, remove whitespace. Only exact frozen alias matches. Long/descriptive output can be valid semantically and will be available for review.',
        'ablation': 'Every swap has the same question text and differs in gold answer. Report accuracy vs original gold, accuracy vs displayed image gold, and paired both-correct rate. No-image omits image placeholder tokens.',
        'comparisons': ['reproduced vs official on original image', 'short vs baseline on original image',
                        'original vs swapped (original gold)', 'original vs no-image'],
        'uncertainty': 'Seed1726; 2000 percentile bootstrap replicates resampling whole image pairs. Descriptive, reported separately for synthetic and natural/derived images.',
    }
    write(HERE / 'protocol.json', protocol)
    print('FROZEN', len(rows), sha(HERE / 'protocol.json'), flush=True)

if __name__ == '__main__':
    main()
