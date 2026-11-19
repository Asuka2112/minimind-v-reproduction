"""One server task named 评测, with GPU occupancy checks and resumable predictions."""
import argparse
import ctypes
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from prepare import HERE, MODELS, TRAIN, read_rows, sha, write


def now():
    return datetime.now(timezone.utc).isoformat()


def set_name():
    if sys.platform == 'linux':
        libc = ctypes.CDLL(None)
        libc.prctl(15, ctypes.c_char_p('评测'.encode()), 0, 0, 0)


def load_protocol():
    protocol = json.loads((HERE/'protocol.json').read_text())
    for filename, digest in protocol['code_sha256'].items():
        assert sha(HERE/filename) == digest, filename
    assert sha(HERE/'data/questions.jsonl') == protocol['dataset_sha256']
    return protocol


def worker(label):
    protocol = load_protocol()
    protocol_sha = sha(HERE/'protocol.json')
    rows = read_rows(HERE/'data/questions.jsonl')
    source = TRAIN/'src/model'
    for filename, digest in protocol['model_source_sha256'].items():
        assert sha(source/filename) == digest, filename
    weight, digest = MODELS[label]
    assert sha(weight) == digest == protocol['models'][label]['checkpoint_sha256']
    for row in rows:
        assert sha(HERE/'data'/row['image']) == row['image_sha256']

    sys.path.insert(0, str(TRAIN/'src'))
    import torch
    from PIL import Image
    from transformers import AutoTokenizer
    from model.model_vlm import MiniMindVLM, VLMConfig

    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = MiniMindVLM(VLMConfig(), vision_model_path=str(source/'siglip2-base-p32-256-ve'))
    assert model.vision_encoder is not None
    result = model.load_state_dict(torch.load(weight, map_location='cpu', weights_only=True), strict=False)
    assert not result.unexpected_keys
    assert set(result.missing_keys) == {key for key in model.state_dict() if key.startswith('vision_encoder.')}
    model = model.float().eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    eos = tokenizer.eos_token_id
    limit = protocol['decode']['max_new_tokens']
    path = HERE/f'predictions/{label}.jsonl'
    path.parent.mkdir(exist_ok=True)
    cases = [(prompt, condition, row) for prompt in protocol['prompts']
             for condition in protocol['conditions'] for row in rows]
    existing = read_rows(path) if path.exists() else []
    assert len(existing) <= len(cases)
    for record, (prompt, condition, row) in zip(existing, cases):
        assert (record['prompt'], record['condition'], record['id']) == (prompt, condition, row['id'])
        assert record['model'] == label and record['protocol_sha256'] == protocol_sha
    started = now()
    print('模型已加载', label, torch.cuda.get_device_name(0), flush=True)
    with torch.inference_mode(), path.open('a', buffering=1) as stream:
        for i, (prompt, condition, row) in enumerate(cases[len(existing):], len(existing)+1):
            question = row['question'] + protocol['prompts'][prompt]
            image_key = {'correct': 'image', 'swapped': 'swapped_image', 'none': None}[condition]
            image_path = row[image_key] if image_key else None
            pixels = None
            if image_path:
                with Image.open(HERE/'data'/image_path) as image:
                    pixels = {k: v.cuda() for k, v in model.image2tensor(image.convert('RGB'), model.processor).items()}
                question = model.config.image_special_token * 64 + '\n' + question
            prompt_text = tokenizer.apply_chat_template([{'role': 'user', 'content': question}],
                                                        tokenize=False, add_generation_prompt=True, open_thinking=False)
            encoded = tokenizer(prompt_text, return_tensors='pt', add_special_tokens=False)
            ids = encoded.input_ids.cuda()
            assert (ids == model.config.image_ids[0]).sum().item() == (64 if pixels else 0)
            t0 = time.monotonic()
            tokens = model.generate(inputs=ids, pixel_values=pixels, max_new_tokens=limit, logits_to_keep=1,
                                    eos_token_id=eos, do_sample=False, top_k=0, top_p=1.0, temperature=1.0,
                                    repetition_penalty=1.0)[0, ids.shape[1]:].tolist()
            prediction = tokenizer.decode(tokens, skip_special_tokens=True)
            record = {'id': row['id'], 'model': label, 'prompt': prompt, 'condition': condition,
                      'protocol_sha256': protocol_sha, 'image': image_path, 'prediction': prediction,
                      'generated_ids': tokens, 'hit_length_cap': len(tokens)==limit and tokens[-1]!=eos,
                      'seconds': time.monotonic()-t0}
            stream.write(json.dumps(record, ensure_ascii=False)+'\n')
            if i == 1 or i % 20 == 0 or i == len(cases):
                write(HERE/f'progress_{label}.json', {'name': '评测', 'model': label, 'done': i, 'total': len(cases), 'at': now()})
    write(HERE/f'receipt_{label}.json', {'model': label, 'started_at': started, 'finished_at': now(),
                                       'rows': len(cases), 'sha256': sha(path), 'gpu': os.environ['CUDA_VISIBLE_DEVICES'],
                                       'protocol_sha256': protocol_sha, 'checkpoint_sha256': digest})
    print('推理完成', label, len(cases), flush=True)


def acquire_gpus():
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,uuid',
                                         '--format=csv,noheader,nounits'], text=True)
    occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                        '--format=csv,noheader,nounits'], text=True)
    occupied = {line.split(',')[0].strip() for line in occupied.splitlines()}
    locks = Path.home()/'.gpu_leases'
    locks.mkdir(exist_ok=True)
    leases = {}
    for line in inventory.splitlines():
        index, memory, uuid = [field.strip() for field in line.split(',')]
        if int(memory)>256 or uuid in occupied:
            continue
        lease = (locks/f'gpu-{index}.lock').open('a')
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lease.close()
            continue
        leases[index] = lease
        if len(leases) == len(MODELS):
            return leases
    for lease in leases.values():
        lease.close()
    raise RuntimeError('空闲 GPU 不足，未启动模型进程。')


def controller():
    from score import main as score

    with (HERE/'task.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = HERE/'status.json'
        if state_path.exists() and json.loads(state_path.read_text())['status']=='complete':
            print('评测已完成。')
            return
        started = now()
        state = {'name': '评测', 'status': 'starting', 'pid': os.getpid(), 'started_at': started}
        write(state_path, state)
        processes = {}
        leases = {}
        try:
            protocol = load_protocol()
            leases = acquire_gpus()
            for model, gpu in zip(MODELS, leases):
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=gpu, HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                           OMP_NUM_THREADS='2', TMPDIR=str(HERE/'tmp'))
                with (HERE/f'{model}.log').open('a') as logfile:
                    command = ['./runtime/bin/python', '-u', 'task.py', '--worker', model]
                    processes[model] = subprocess.Popen(command, env=env, cwd=HERE, stdout=logfile, stderr=subprocess.STDOUT)
            state.update(status='running', protocol_sha256=sha(HERE/'protocol.json'),
                         workers={m: {'pid': p.pid, 'gpu': gpu} for (m,p),gpu in zip(processes.items(),leases)})
            write(state_path, state)
            while any(p.poll() is None for p in processes.values()):
                if any(p.poll() not in (None, 0) for p in processes.values()):
                    raise RuntimeError('子任务失败，请查看模型日志。')
                time.sleep(3)
            exits = {m: p.wait() for m,p in processes.items()}
            assert all(code==0 for code in exits.values()), exits
            records = []
            for model in MODELS:
                path = HERE/f'predictions/{model}.jsonl'
                receipt = json.loads((HERE/f'receipt_{model}.json').read_text())
                assert sha(path) == receipt['sha256']
                records.extend(read_rows(path))
            report = score(protocol, records, started)
            state.update(status='complete', finished_at=now(), worker_exit_codes=exits,
                         predictions=report['prediction_count'], results_sha256=sha(HERE/'results.json'))
            write(state_path, state)
            print('评测完成', report['prediction_count'], flush=True)
        except BaseException:
            for process in processes.values():
                if process.poll() is None:
                    process.terminate()
            for process in processes.values():
                process.wait()
            state.update(status='failed', finished_at=now(), error=traceback.format_exc(),
                         worker_exit_codes={m: p.returncode for m,p in processes.items()})
            write(state_path, state)
            raise
        finally:
            for lease in leases.values():
                lease.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', choices=tuple(MODELS))
    args = parser.parse_args()
    set_name()
    if args.worker:
        worker(args.worker)
    else:
        controller()
