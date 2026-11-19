"""Record original trainer progress without changing its objective or optimizer."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
CONFIG = json.loads((ROOT / 'training/config.json').read_text())
sys.path.insert(0, str(ROOT / 'src'))
import torch
import torch.distributed as dist


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def record(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps({'at': now(), **value}, ensure_ascii=False) + '\n')


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def mean_scalar(tensor):
    value = tensor.detach().float().clone()
    if dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= dist.get_world_size()
    return value.item()


def install(stage, stop_after=None):
    from trainer import trainer_utils as utils
    run = ROOT / 'runs' / stage
    run.mkdir(parents=True, exist_ok=True)
    original_logger, original_checkpoint = utils.Logger, utils.vlm_checkpoint
    original_init = utils.init_vlm_model
    started = time.monotonic()

    def logger(message):
        original_logger(message)
        if utils.is_main_process():
            entry = {'event': 'log', 'message': str(message)}
            match = re.search(r'Epoch:\[(\d+)/(\d+)\]\((\d+)/(\d+)\), loss: ([\d.e+-]+)', str(message))
            if match:
                epoch, epochs, step, steps, loss = match.groups()
                entry.update(epoch=int(epoch), epochs=int(epochs), step=int(step), steps=int(steps), loss=float(loss))
                dump(run / 'progress.json', {'status': 'running', **entry, 'at': now()})
            record(run / 'events.jsonl', entry)

    def checkpoint(config, *args, **kwargs):
        saving = kwargs.get('model') is not None
        if saving:
            folder = Path(kwargs.get('save_dir', '../checkpoints'))
            name = kwargs.get('weight', 'pretrain_vlm')
            current = folder / f'{name}_{config.hidden_size}_resume.pth'
            if current.is_file():
                previous = current.with_name(current.stem + '.previous.pth')
                previous.unlink(missing_ok=True)
                os.link(current, previous)
            kwargs['provenance'] = {'source_commit': CONFIG['source_commit'], 'stage': stage,
                                    'config': CONFIG, 'at': now(), 'world_size': dist.get_world_size() if dist.is_initialized() else 1}
        result = original_checkpoint(config, *args, **kwargs)
        if saving:
            record(run / 'events.jsonl', {'event': 'checkpoint', 'epoch': kwargs['epoch'], 'step': kwargs['step']})
            if stop_after is not None and kwargs['step'] >= stop_after:
                raise SystemExit(75)
        elif result is not None:
            provenance = result.get('provenance', {})
            if provenance.get('config') != CONFIG or provenance.get('stage') != stage:
                raise RuntimeError('Resume checkpoint provenance differs from this fixed pipeline configuration')
            record(run / 'events.jsonl', {'event': 'resume_loaded', 'epoch': result['epoch'], 'step': result['step']})
        return result

    def initialize(*args, **kwargs):
        result = original_init(*args, **kwargs)
        model = result[0]
        if model.vision_encoder is None:
            raise RuntimeError('Vision encoder missing; refusing a text-only training run')
        if utils.is_main_process():
            dump(run / 'initialization.json', {'at': now(), 'total_parameters': sum(p.numel() for p in model.parameters()),
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'trainable_names': [name for name, p in model.named_parameters() if p.requires_grad],
                'parameter_hashes': {name: tensor_sha(p) for name, p in model.named_parameters()}})
        return result

    utils.Logger, utils.vlm_checkpoint, utils.init_vlm_model = logger, checkpoint, initialize
    return started
