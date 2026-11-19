"""Prepare an isolated copy of the pinned upstream code and verified resources."""
import difflib
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from instrument import ROOT, CONFIG, dump, now


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


if __name__ == '__main__':
    import pyarrow.parquet as pq
    import pyarrow as pa
    import torch
    from PIL import Image
    from transformers import AutoTokenizer
    for name in ['artifacts', 'logs', 'data', 'weights', 'runs', 'exports', 'evaluation', 'tmp', 'cache']:
        (ROOT / name).mkdir(exist_ok=True)
    if not (ROOT / 'src').exists():
        shutil.copytree(ROOT / 'upstream', ROOT / 'src')
    diffs = []
    # All-rank save barriers prevent rank 0 from racing the next DDP forward.
    for name in ['train_pretrain_vlm.py', 'train_sft_vlm.py']:
        source = ROOT / 'upstream/trainer' / name
        original = source.read_text()
        text = original.replace("warnings.filterwarnings('ignore')", "from instrument import mean_scalar\n\nwarnings.filterwarnings('ignore')")
        text = text.replace('        scaler.scale(loss).backward()',
            "        if not torch.isfinite(loss).all():\n            raise FloatingPointError('Non-finite training loss')\n        scaler.scale(loss).backward()")
        text = text.replace('current_loss = loss.item() * args.accumulation_steps', 'current_loss = mean_scalar(loss * args.accumulation_steps)')
        text = text.replace('current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0',
                            'current_aux_loss = mean_scalar(res.aux_loss) if res.aux_loss is not None else 0.0')
        text = text.replace('            torch.save(clean_state_dict, ckp)',
                            "            torch.save(clean_state_dict, ckp + '.tmp')\n            os.replace(ckp + '.tmp', ckp)")
        start = text.index('        if (step % args.save_interval')
        end = text.index('\n        del input_ids', start)
        block = text[start:end]
        first, body = block.split('\n', 1)
        first = first.replace(' and is_main_process()', '')
        text = text[:start] + first + '\n            if dist.is_initialized(): dist.barrier()\n            if is_main_process():\n' + '\n'.join('    ' + line if line.strip() else line for line in body.splitlines()) + '\n            if dist.is_initialized(): dist.barrier()\n' + text[end:]
        compile(text, name, 'exec')
        (ROOT / 'src/trainer' / name).write_text(text)
        diffs.extend(difflib.unified_diff(original.splitlines(True), text.splitlines(True), fromfile='upstream/trainer/' + name, tofile='src/trainer/' + name))
    eval_source = ROOT / 'upstream/eval_vlm.py'
    eval_original = eval_source.read_text()
    eval_text = eval_original.replace("ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'",
        "ckp = os.path.join(args.save_dir, f'{args.weight}_{args.hidden_size}{moe_suffix}.pth')")
    (ROOT / 'src/eval_vlm.py').write_text(eval_text)
    (ROOT / 'artifacts/trainer.patch').write_text(''.join(diffs))
    (ROOT / 'artifacts/eval_path.patch').write_text(''.join(difflib.unified_diff(eval_original.splitlines(True), eval_text.splitlines(True), fromfile='upstream/eval_vlm.py', tofile='src/eval_vlm.py')))
    resources = CONFIG['resources']
    vision = Path(resources['vision']['existing_path'])
    assert sha(vision / 'model.safetensors') == resources['vision']['sha256']
    assert sha(ROOT / 'weights/llm_768.pth') == resources['language']['sha256']
    vision_link = ROOT / 'src/model/siglip2-base-p32-256-ve'
    if not vision_link.exists(): vision_link.symlink_to(vision, target_is_directory=True)
    raw_sft = Path(resources['dataset']['existing_sft'])
    sft_link = ROOT / 'data/sft_i2t.parquet'
    if not sft_link.exists(): sft_link.symlink_to(raw_sft)
    manifests = {}
    for stage in CONFIG['stages']:
        path = ROOT / 'data' / stage['data']
        actual = sha(path)
        assert actual == resources['dataset'][stage['name'] + '_sha256'], (path, actual)
        parquet = pq.ParquetFile(path)
        manifests[stage['name']] = {'path': str(path), 'resolved_path': str(path.resolve()), 'sha256': actual,
            'bytes': path.stat().st_size, 'rows': parquet.metadata.num_rows, 'row_groups': parquet.num_row_groups,
            'schema': str(parquet.schema_arrow), 'training_filtering': 'none: complete pinned official file'}
    # Strictly verify the official language checkpoint before adding a random projector.
    sys.path.insert(0, str(ROOT / 'src'))
    importlib.invalidate_caches()
    from model.model_vlm import MiniMindVLM, VLMConfig
    state = torch.load(ROOT / 'weights/llm_768.pth', map_location='cpu', weights_only=True)
    model = MiniMindVLM(VLMConfig(), vision_model_path=str(vision))
    result = model.load_state_dict(state, strict=False)
    assert not result.unexpected_keys
    assert all(name.startswith('vision_encoder.') or name.startswith('vision_proj.') for name in result.missing_keys)
    assert all(torch.isfinite(t).all() for t in state.values())
    dump(ROOT / 'artifacts/language_load.json', {'missing_keys': result.missing_keys, 'unexpected_keys': result.unexpected_keys,
        'language_tensors': len(state), 'sha256': resources['language']['sha256'], 'status': 'verified'})
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'src/model', local_files_only=True)
    from dataset.lm_dataset import VLMDataset
    # A small genuine upstream-data smoke cohort, never presented as full training.
    parquet = pq.ParquetFile(ROOT / 'data/sft_i2t.parquet')
    rows = []
    for batch in parquet.iter_batches(batch_size=16):
        for row in batch.to_pylist():
            conversations = json.loads(row['conversations']) if isinstance(row['conversations'], str) else row['conversations']
            if row['image_bytes'] and any(turn.get('role') == 'assistant' and turn.get('content') for turn in conversations):
                rows.append(row)
            if len(rows) == 8: break
        if len(rows) == 8: break
    smoke_path = ROOT / 'data/smoke.parquet'
    pq.write_table(pa.Table.from_pylist(rows, schema=parquet.schema_arrow), smoke_path)
    smoke = VLMDataset(str(smoke_path), tokenizer, preprocess=model.processor, max_length=768)
    smoke_checks = []
    for index in range(len(smoke)):
        ids, labels, pixels = smoke[index]
        assert (labels != -100).any()
        assert pixels['pixel_values'].shape[-3:] == (3, 256, 256)
        smoke_checks.append({'row': index, 'target_tokens': int((labels != -100).sum()), 'image_marker_tokens': int((ids == 12).sum())})
    dump(ROOT / 'artifacts/data_manifest.json', {'at': now(), 'datasets': manifests, 'smoke_checks': smoke_checks,
        'audit_scope': 'Full-file SHA256/schema/row counts; eight processed training examples. No claim of exhaustive manual label review.'})
    for name in ['smoke', 'ddp_smoke', 'pretrain', 'sft']:
        folder = ROOT / 'runs' / name
        for sub in ['trainer', 'out', 'checkpoints']:
            (folder / sub).mkdir(parents=True, exist_ok=True)
        if not (folder / 'model').exists(): (folder / 'model').symlink_to(ROOT / 'src/model', target_is_directory=True)
        if name in ['smoke', 'ddp_smoke', 'pretrain'] and not (folder / 'out/llm_768.pth').exists():
            (folder / 'out/llm_768.pth').symlink_to(ROOT / 'weights/llm_768.pth')
    pretrain_link = ROOT / 'runs/sft/out/pretrain_vlm_768.pth'
    if not pretrain_link.is_symlink(): pretrain_link.symlink_to(ROOT / 'runs/pretrain/out/pretrain_vlm_768.pth')
    freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    (ROOT / 'artifacts/environment.lock.txt').write_text(freeze)
    dump(ROOT / 'artifacts/bootstrap.json', {'at': now(), 'status': 'ready', 'config': CONFIG,
        'source_manifest': {str(p.relative_to(ROOT / 'upstream')): sha(p) for p in sorted((ROOT / 'upstream').rglob('*')) if p.is_file()},
        'python': sys.version, 'torch': torch.__version__, 'patch_sha256': sha(ROOT / 'artifacts/trainer.patch'),
        'eval_patch_sha256': sha(ROOT / 'artifacts/eval_path.patch')})
    print(json.dumps({'status': 'ready', 'datasets': manifests, 'smoke_examples': len(smoke_checks)}, ensure_ascii=False, indent=2))
