"""Check label geometry, split isolation and image integrity before freezing."""
from collections import Counter
import hashlib
import random

from PIL import Image, ImageDraw, ImageFont, ImageOps

from common import HERE, PRIOR, dump, load, now, read_rows, sha
from build_balanced import PALETTE

DATA = HERE / 'data'
SIZES = {'control': 3200, 'targeted': 3200, 'dev': 80, 'test': 240, 'challenge': 240}
NUMBER_LABELS = {'0', '1', '2', '3', '4', '5+'}


def check_geometry(row):
    if row.get('subtask') == 'marked_color':
        x1,y1,x2,y2 = row['color_box']
        assert 0 <= x1 < x2 < 448 and 0 <= y1 < y2 < 448
        with Image.open(DATA / row['image']) as im:
            pixel = im.getpixel(((x1+x2)//2,(y1+y2)//2))
        distances = {name:sum((a-b)**2 for a,b in zip(pixel,rgb)) for name,rgb in PALETTE.items()}
        assert min(distances,key=distances.get) == row['answer'], row['id']
        return
    if row['task'] == 'count':
        items = row['instance_boxes']
        counts = row['instance_counts']
        seen = Counter(str(item['cls']) for item in items)
        assert set(seen) <= counts.keys()
        assert all(seen[key] == count for key, count in counts.items())
        assert len(row['donors']) == len(items)
        target, distractor = counts.values()
        if row['subtask'] == 'number':
            answer = '5+' if target >= 5 else str(target)
        else:
            answer = '更多' if target > distractor else '更少' if target < distractor else '一样多'
        assert row['answer'] == answer, row['id']
        for i, item in enumerate(items):
            x1, y1, x2, y2 = item['box']
            assert 0 <= x1 < x2 < 448 and 0 <= y1 < y2 < 448
            for other in items[:i]:
                a, b, c, d = other['box']
                assert x2 <= a or c <= x1 or y2 <= b or d <= y1
        for occluded in row['occlusions']:
            assert .02 <= occluded['fraction'] <= .18
            assert occluded['annotation'] in {aid for _, aid in row['donors']}
        return
    if row['task'] != 'position':
        return
    blue, red = row['blue_box'], row['red_box']
    for x1, y1, x2, y2 in (blue, red):
        assert 0 <= x1 < x2 < 448 and 0 <= y1 < y2 < 448
    axis = row['subtask']
    if axis == 'depth':
        x = (max(blue[0], red[0]) + min(blue[2], red[2])) // 2
        y = (max(blue[1], red[1]) + min(blue[3], red[3])) // 2
        assert all(box[0] < x < box[2] and box[1] < y < box[3] for box in (blue, red))
        with Image.open(DATA / row['image']) as im:
            r, _, b = im.getpixel((x, y))
        answer = '前面' if b > r + 120 else '后面'
        assert row['draw_order'][-1] == ('blue' if answer == '前面' else 'red')
    else:
        n = 0 if axis == 'horizontal' else 1
        assert axis in ('horizontal', 'vertical')
        first = blue[n + 2] < red[n]
        assert first or red[n + 2] < blue[n]
        answer = ('左边' if first else '右边') if n == 0 else ('上方' if first else '下方')
    assert row['answer'] == answer, row['id']


def check_pairs(rows):
    one = {r['id']: r for r in rows}
    pairs = {r['pair'] for r in rows}
    assert 2 * len(pairs) == len(rows)
    for row in rows:
        other = one[row['swapped_id']]
        assert other['pair'] == row['pair'] and other['swapped_id'] == row['id']
        assert other['question'] == row['question'] and other['task'] == row['task']
        assert other['answer'] != row['answer'] and other['image_sha256'] != row['image_sha256']
    return len(pairs)


def panels(train, challenge):
    output = HERE / 'qa'
    output.mkdir(exist_ok=True)
    rng = random.Random(2916)
    cohorts = {
        'train-natural': [r for r in train if r['task'] in ('object','color')],
        'train-count': [r for r in train if r['task'] == 'count'],
        'train-position': [r for r in train if r['task'] == 'position'],
        'challenge': challenge,
    }
    selected = {}
    for name, rows in cohorts.items():
        keys = sorted({(r['subtask'], r['answer'] if r['task'] in ('count','position') else '') for r in rows})
        sample = []
        for sub, answer in keys:
            subset = [r for r in rows if r['subtask'] == sub and (not answer or r['answer'] == answer)]
            sample.extend(rng.sample(subset, min(len(subset), 12 if name == 'train-natural' else 2)))
        assert len(sample) <= 30
        panel = Image.new('RGB', (180 * 6, 198 * ((len(sample)+5)//6)), 'white')
        pen = ImageDraw.Draw(panel)
        font = ImageFont.load_default()
        for i, row in enumerate(sample):
            x, y = i % 6 * 180, i // 6 * 198
            with Image.open(DATA / row['image']) as im:
                panel.paste(ImageOps.contain(im.convert('RGB'), (176,176)), (x+2,y+2))
            pen.text((x+5,y+181), f'{i+1:02d} {row["id"].removeprefix("challenge-")}', fill='#181818', font=font)
        panel.save(output / f'{name}.jpg', quality=90)
        selected[name] = [{k:r[k] for k in ('id','image','question','answer','subtask')} for r in sample]
    dump(output / 'sample.json', selected)
    return {k:len(v) for k,v in selected.items()}


def main():
    assert not (HERE/'protocol.json').exists()
    manifest = load(DATA/'manifest.json')
    catalog = HERE/'donor_qa/index.json'
    approvals = HERE/'donor_qa/approved.json'
    assert sha(catalog) == manifest['donor_gallery_index_sha256']
    assert sha(approvals) == manifest['donor_selection_sha256']
    candidates = load(catalog)
    allowed = {name:set() for name in ('train','challenge')}
    for key, indices in load(approvals).items():
        allowed[key.split('-')[0]].update(tuple(candidates[key][i-1]) for i in indices)
    for file, expected in manifest['files'].items():
        assert sha(DATA/file) == expected, file
    rows = {name:read_rows(DATA/f'{name}.jsonl') for name in SIZES}
    assert {k:len(v) for k,v in rows.items()} == SIZES
    train, control = rows['targeted'], rows['control']
    for a,b in zip(control,train):
        assert all(a[k]==b[k] for k in ('id','image','image_sha256','donors'))
    assert sum(a['question'] != b['question'] for a,b in zip(control,train)) == manifest['changed_questions']
    shared = train + rows['dev'] + rows['test'] + rows['challenge']
    assert len({r['id'] for r in shared}) == len({r['image_sha256'] for r in shared}) == len(shared)
    for r in shared:
        path = DATA / r['image']
        assert sha(path) == r['image_sha256']
        with Image.open(path) as im:
            im.verify()
    for name in ('dev','test','text_regression'):
        assert sha(DATA/f'{name}.jsonl') == sha(PRIOR/f'data/{name}.jsonl')
    for r in train + rows['challenge']:
        check_geometry(r)
        if r['task'] in ('count','position'):
            donors = allowed['train' if not r['id'].startswith('challenge-') else 'challenge']
            assert {tuple(donor) for donor in r['donors']} <= donors
    for r in control + train:
        assert r['answer'] in r['aliases'] and 0 < len(r['answer']) <= 12
    def donor_ids(records):
        return {i for r in records for i, _ in r['donors']}
    old = read_rows(PRIOR/'data/control.jsonl') + rows['dev'] + rows['test']
    old_ids = {r['coco_id'] for r in old}
    assert donor_ids(rows['challenge']).isdisjoint(old_ids)
    assert donor_ids(train).isdisjoint(donor_ids(rows['challenge']))
    assert donor_ids(train) <= {r['coco_id'] for r in old[:-320]}
    counts = Counter(r['answer'] for r in train if r.get('subtask') == 'number')
    assert set(counts) == NUMBER_LABELS and set(counts.values()) == {140}
    fresh = Counter(r['answer'] for r in rows['challenge'] if r.get('subtask') == 'number')
    assert set(fresh) == NUMBER_LABELS and set(fresh.values()) == {16}
    pair_counts = {name: check_pairs(rows[name]) for name in ('dev','test','challenge')}
    assert pair_counts == {'dev':40, 'test':120, 'challenge':120}
    result = {'pass':True, 'checked_at':now(), 'manifest_sha256':sha(DATA/'manifest.json'),
              'sizes':SIZES, 'verified_distinct_image_bytes':len(shared),
              'verified_count_and_position_rows':sum(r['task'] in ('count','position') for r in train+rows['challenge']),
              'verified_color_rows':sum(r.get('subtask')=='marked_color' for r in train),
              'train_number_labels':counts, 'challenge_number_labels':fresh, 'pairs':pair_counts,
              'changed_questions':manifest['changed_questions'], 'visual_sample_counts':panels(train, rows['challenge'])}
    dump(HERE/'quality.json',result)
    print(result,flush=True)


if __name__ == '__main__':
    main()
