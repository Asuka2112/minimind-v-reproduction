"""Reaudit actual complete fine-tuning sources; never use an older held-out split."""
import collections
import concurrent.futures
import hashlib
import io
import sqlite3
import pyarrow.parquet as pq
from PIL import Image
from common import *

def decode(item):
    key, raw = item
    with Image.open(io.BytesIO(raw)) as image:
        image = image.convert('RGB')
        rgb = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        values = list(image.resize((9, 8)).convert('L').getdata())
        dh = sum(int(values[y*9+x] > values[y*9+x+1]) << (y*8+x) for y in range(8) for x in range(8))
    return key, (rgb, dh)

def main():
    test = rows(HERE / 'data/mmstar.jsonl')
    flags = {x['image_sha256']: {'raw_exact': False, 'rgb_exact': False, 'near_candidate': False, 'nearest_distance': None} for x in test}
    raw_test = collections.defaultdict(set)
    buckets = [collections.defaultdict(set) for _ in range(4)]
    hashes = {x['image_sha256']: x['dhash'] for x in test}
    for x in test:
        raw_test[x['image_file_sha256']].add(x['image_sha256'])
    for key, dh in hashes.items():
        for b in range(4): buckets[b][(dh >> (16*b)) & 65535].add(key)
    c = sqlite3.connect(f'file:{OLD}/data/image_index.sqlite3?mode=ro', uri=True)
    cache = {raw: ((rgb, int(dh, 16)) if valid else None) for raw, rgb, dh, valid in c.execute('SELECT raw_sha,rgb_sha,dhash,valid FROM image')}
    c.close()
    old_keys = set(cache)
    marked = set()
    def match(key):
        if key in marked: return
        marked.add(key)
        for image in raw_test.get(key, []): flags[image]['raw_exact'] = True
        rgb, dh = cache[key]
        if rgb in flags: flags[rgb]['rgb_exact'] = True
        candidates = set()
        for b in range(4): candidates.update(buckets[b][(dh >> (16*b)) & 65535])
        for image in candidates:
            distance = (dh ^ hashes[image]).bit_count()
            if distance <= 3:
                flag = flags[image]
                flag['near_candidate'] = True
                if flag['nearest_distance'] is None or distance < flag['nearest_distance']: flag['nearest_distance'] = distance
    sources = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for name, expected in [('sft', '712f4026cd0e21b369feddca7334b1e465cb8182b5f298006f3f4f877f926643'),
                               ('pretrain', '65761f37d1947d54a1d85457ff70938275e4ef58ba5cedcd02463a3a247c93fd')]:
            path = TRAIN / f'data/{name}_i2t.parquet'
            assert sha(path) == expected
            pf = pq.ParquetFile(path)
            seen = set(); count = 0; decoded = 0; missing_index = 0
            for batch in pf.iter_batches(batch_size=4096, columns=['image_bytes']):
                unknown = {}
                for raw in batch.column(0).to_pylist():
                    assert isinstance(raw, bytes)
                    key = hashlib.sha256(raw).hexdigest()
                    if key not in seen:
                        seen.add(key)
                        if key not in old_keys: missing_index += 1
                        if cache.get(key) is None: unknown[key] = raw
                        else: match(key)
                for key, info in pool.map(decode, unknown.items()):
                    cache[key] = info; decoded += 1; match(key)
                count += len(batch)
                if count % (4096*20) == 0:
                    dump(HERE / 'artifacts/overlap_progress.json', {'at': now(), 'source': name, 'rows': count, 'total': pf.metadata.num_rows, 'newly_decoded': decoded})
            assert count == pf.metadata.num_rows
            if name == 'sft': assert missing_index == 0, 'Existing SFT index coverage incomplete'
            sources.append({'name': name, 'path': str(path), 'sha256': expected, 'rows': count,
                            'unique_encoded_images': len(seen), 'absent_from_old_full_sft_index': missing_index, 'newly_decoded': decoded})
    for x in test:
        flag = flags[x['image_sha256']]
        x['exclude_exact'] = flag['raw_exact'] or flag['rgb_exact']
        x['exclude_conservative'] = x['exclude_exact'] or flag['near_candidate']
    summary = {'at': now(), 'status': 'complete', 'protocol_sha256': sha(HERE / 'protocol.json'), 'sources': sources,
               'rows': len(test), 'unique_rgb_images': len(flags), 'exact_match_rows': sum(x['exclude_exact'] for x in test),
               'conservative_match_rows': sum(x['exclude_conservative'] for x in test),
               'exact_free_rows': sum(not x['exclude_exact'] for x in test), 'conservative_free_rows': sum(not x['exclude_conservative'] for x in test),
               'scope': 'Both actual full official fine-tuning sources; direct dHash<=3 is a conservative candidate filter, not confirmed semantic duplication. Foundation pretraining contamination remains unknown.',
               'flags': flags, 'row_masks': {x['id']: {'exact_free': not x['exclude_exact'], 'conservative_free': not x['exclude_conservative']} for x in test}}
    dump(HERE / 'artifacts/overlap_audit.json', summary)
    print('OVERLAP_COMPLETE', {k: summary[k] for k in ['rows', 'exact_match_rows', 'conservative_match_rows']}, flush=True)

if __name__ == '__main__': main()
