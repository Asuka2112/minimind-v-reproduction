"""Audit real checkpoints, export via upstream converter, and run fixed image demos."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

from instrument import ROOT, CONFIG, dump, now, tensor_sha
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import save_file
from model.model_vlm import MiniMindVLM, VLMConfig
from model.model_minimind import precompute_freqs_cis


def file_sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_native(path):
    model = MiniMindVLM(VLMConfig(), vision_model_path=str(ROOT / 'src/model/siglip2-base-p32-256-ve'))
    state = torch.load(path, map_location='cpu', weights_only=True)
    result = model.load_state_dict(state, strict=False)
    if 'llm_' in path.name:
        assert all(k.startswith(('vision_encoder.', 'vision_proj.')) for k in result.missing_keys)
    else:
        assert all(k.startswith('vision_encoder.') for k in result.missing_keys)
    assert not result.unexpected_keys
    return model.eval()


def model_inputs(model, tokenizer, image_path):
    prompt = '请描述这张图中的主要物体和场景。'
    text = tokenizer.apply_chat_template([{'role': 'user', 'content': model.config.image_special_token * 64 + '\n' + prompt}],
                                         tokenize=False, add_generation_prompt=True, open_thinking=False)
    inputs = tokenizer(text, return_tensors='pt')
    with Image.open(image_path) as image:
        pixels = dict(MiniMindVLM.image2tensor(image.convert('RGB'), model.processor))
    return inputs, pixels


def audit_checkpoint(stage):
    run = ROOT / 'runs' / stage
    prefix = {'smoke': 'smoke', 'ddp_smoke': 'ddp_smoke', 'pretrain': 'pretrain_vlm', 'sft': 'sft_vlm'}[stage]
    path = run / 'checkpoints' / f'{prefix}_768_resume.pth'
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    assert all(torch.isfinite(t).all() for t in checkpoint['model'].values())
    steps = []
    for state in checkpoint['optimizer']['state'].values():
        steps.append(int(state['step'].item()) if hasattr(state['step'], 'item') else int(state['step']))
        assert all(not isinstance(t, torch.Tensor) or torch.isfinite(t).all() for t in state.values())
    initial = json.loads((run / 'initialization.json').read_text())
    trainable = set(initial['trainable_names'])
    changed, frozen_changed = [], []
    for name, initial_sha in initial['parameter_hashes'].items():
        if tensor_sha(checkpoint['model'][name]) != initial_sha:
            (changed if name in trainable else frozen_changed).append(name)
    assert changed, 'No trainable parameter updated'
    assert not frozen_changed, frozen_changed
    if stage == 'smoke':
        assert min(steps) == max(steps) == 8, steps
        events = [json.loads(line) for line in (run / 'events.jsonl').read_text().splitlines()]
        assert any(x['event'] == 'resume_loaded' and x['step'] == 2 for x in events)
    elif stage == 'ddp_smoke':
        assert min(steps) == max(steps) == 2, steps
        assert all(json.loads((run / f'receipt-success-rank{rank}.json').read_text())['exit_code'] == 0 for rank in range(len(CONFIG['gpus'])))
    else:
        manifest = json.loads((ROOT / 'artifacts/data_manifest.json').read_text())['datasets'][stage]
        cfg = next(x for x in CONFIG['stages'] if x['name'] == stage)
        world = len(CONFIG['gpus'])
        per_rank_samples = (manifest['rows'] + world - 1) // world
        steps_per_epoch = (per_rank_samples + cfg['batch_size'] - 1) // cfg['batch_size']
        expected = steps_per_epoch * cfg['epochs']
        assert min(steps) == max(steps) == expected, (min(steps), max(steps), expected)
        assert checkpoint['epoch'] == cfg['epochs'] - 1 and checkpoint['step'] == steps_per_epoch
        receipts = [json.loads((run / f'receipt-success-rank{rank}.json').read_text()) for rank in range(world)]
        assert all(x['exit_code'] == 0 for x in receipts)
    result = {'status': 'verified', 'at': now(), 'stage': stage, 'resume_sha256': file_sha(path),
              'epoch': checkpoint['epoch'], 'step_in_epoch': checkpoint['step'],
              'adam_step_min': min(steps), 'adam_step_max': max(steps), 'optimizer_parameter_states': len(steps),
              'changed_trainable_tensors': changed, 'changed_frozen_tensors': frozen_changed,
              'trainable_parameters': initial['trainable_parameters'],
              'resume_limit': 'Native epoch/step/model/optimizer/scaler restored. Upstream worker/augmentation RNG is not serialized; not bitwise-continuation evidence.'}
    dump(run / 'checkpoint_audit.json', result)
    return result


def export(stage):
    prefix = stage if stage in ['smoke', 'ddp_smoke'] else stage + '_vlm'
    path = ROOT / 'runs' / stage / 'out' / f'{prefix}_768.pth'
    folder = ROOT / 'exports' / stage
    spec = importlib.util.spec_from_file_location('upstream_converter', ROOT / 'src/scripts/convert_vlm.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.lm_config = VLMConfig(hidden_size=768, num_hidden_layers=8, max_seq_len=8192, use_moe=False)
    previous = Path.cwd()
    try:
        os.chdir(ROOT / 'src/scripts')
        module.convert_torch2transformers_minimind(str(path), str(folder), dtype=torch.float16)
    finally:
        os.chdir(previous)
    # Retain upstream's .bin export, and provide safetensors for the older server torch runtime.
    # This uses the supported HF format rather than disabling Transformers' version safety check.
    exported_state = torch.load(folder / 'pytorch_model.bin', map_location='cpu', weights_only=True)
    save_file({key: value.detach().cpu().contiguous().clone() for key, value in exported_state.items()},
              str(folder / 'model.safetensors'), metadata={'format': 'pt'})
    del exported_state
    native = load_native(path)
    exported = AutoModelForCausalLM.from_pretrained(folder, trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.float32, use_safetensors=True).eval()
    exported.vision_encoder, exported.processor = MiniMindVLM.get_vision_model(str(ROOT / 'src/model/siglip2-base-p32-256-ve'))
    cos, sin = precompute_freqs_cis(dim=exported.config.head_dim, end=exported.config.max_position_embeddings,
                                  rope_base=exported.config.rope_theta, rope_scaling=exported.config.rope_scaling)
    exported.model.freqs_cos, exported.model.freqs_sin = cos, sin
    left, right = native.state_dict(), exported.state_dict()
    assert left.keys() == right.keys()
    assert all(torch.equal(left[key], right[key]) for key in left)
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    inputs, pixels = model_inputs(native, tokenizer, sorted((ROOT / 'src/dataset/eval_images').glob('*.jpg'))[0])
    with torch.inference_mode():
        a = native(input_ids=inputs.input_ids, pixel_values=pixels, logits_to_keep=1).logits
        b = exported(input_ids=inputs.input_ids, pixel_values=pixels, logits_to_keep=1).logits
    delta = (a - b).abs().max().item()
    assert delta == 0, delta
    result = {'at': now(), 'status': 'verified', 'stage': stage, 'format': 'upstream Transformers .bin FP16 plus safetensors',
              'native_weight_sha256': file_sha(path), 'export_weight_sha256': file_sha(folder / 'pytorch_model.bin'),
              'safetensors_sha256': file_sha(folder / 'model.safetensors'),
              'persistent_state_equal': True, 'image_prefill_max_abs_logit_delta': delta,
              'parity_scope': 'Full persistent tensors plus one image prefill against FP16 native artifact reloaded in FP32; not equivalence to full-precision training checkpoint.',
              'loader': 'Attach the pinned external SigLIP encoder and reconstruct FP32 nonpersistent RoPE. local_runtime.py does this through local model construction.'}
    dump(folder / 'export_verification.json', result)
    return result


def evaluate(label, weight):
    torch.manual_seed(CONFIG['seed'])
    model = load_native(weight)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'src/model', local_files_only=True)
    rows = []
    for image in sorted((ROOT / 'src/dataset/eval_images').glob('*.jpg')):
        inputs, pixels = model_inputs(model, tokenizer, image)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        pixels = {k: v.to(device) for k, v in pixels.items()}
        started = time.monotonic()
        with torch.inference_mode():
            result = model.generate(inputs=inputs['input_ids'], attention_mask=inputs['attention_mask'], pixel_values=pixels,
                                    max_new_tokens=CONFIG['evaluation']['max_new_tokens'], do_sample=False, top_k=0, top_p=1.0,
                                    temperature=1.0, repetition_penalty=1.0, eos_token_id=tokenizer.eos_token_id)
        continuation = result[0, inputs['input_ids'].shape[1]:].cpu()
        rows.append({'image': image.name, 'image_sha256': file_sha(image), 'question': '请描述这张图中的主要物体和场景。',
            'prediction': tokenizer.decode(continuation, skip_special_tokens=True), 'generated_tokens': len(continuation),
            'hit_length_limit': len(continuation) >= CONFIG['evaluation']['max_new_tokens'], 'seconds': time.monotonic() - started})
    destination = ROOT / 'evaluation' / label
    destination.mkdir(exist_ok=True)
    (destination / 'predictions.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows))
    dump(destination / 'summary.json', {'at': now(), 'status': 'complete', 'label': label, 'weight_sha256': file_sha(weight),
        'examples': len(rows), 'nonempty': sum(bool(x['prediction'].strip()) for x in rows),
        'hit_length_limit': sum(x['hit_length_limit'] for x in rows), 'decode': CONFIG['evaluation'],
        'scope': 'Upstream six-image qualitative demonstration only. No human labels, no accuracy score, no held-out generalization claim.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['smoke', 'ddp_smoke', 'pretrain', 'sft'])
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--label')
    parser.add_argument('--weight', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.stage:
        audit_checkpoint(args.stage)
        if args.stage in ['pretrain', 'sft'] or args.export: export(args.stage)
    if args.evaluate:
        assert args.label and args.weight
        evaluate(args.label, args.weight)
