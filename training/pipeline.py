"""One durable server process: smoke/resume -> pretrain -> SFT -> export/evaluation."""
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from instrument import ROOT, CONFIG, dump, now, record


def status(state, phase, **extra):
    dump(ROOT / 'pipeline_status.json', {'at': now(), 'status': state, 'phase': phase, 'pid': os.getpid(),
                                        'remote_root': str(ROOT), **extra})


def gpu_check():
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu,uuid', '--format=csv,noheader,nounits'], text=True)
    rows = {}
    for line in text.splitlines():
        index, memory, utilization, uuid = [part.strip() for part in line.split(',')]
        rows[int(index)] = {'memory_mib': int(memory), 'utilization_percent': int(utilization), 'uuid': uuid, 'compute_pids': []}
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
    for line in apps.splitlines():
        if not line.strip(): continue
        uuid, pid = [part.strip() for part in line.split(',')]
        for value in rows.values():
            if value['uuid'] == uuid: value['compute_pids'].append(int(pid))
    for index in CONFIG['gpus']:
        # Utilization is a trailing average and remains nonzero briefly after our own demo exits.
        if rows[index]['memory_mib'] > 256 or rows[index]['compute_pids']:
            raise RuntimeError(f'GPU {index} is occupied; this pipeline will not preempt another job: {rows[index]}')
    dump(ROOT / 'artifacts/last_gpu_check.json', {'at': now(), 'gpus': rows, 'selected': CONFIG['gpus']})


def execute(label, argv, expected=0, cwd=None):
    status('running', label, argv=argv)
    log = ROOT / 'logs' / (label + '.log')
    started = time.monotonic()
    with log.open('a') as stream:
        stream.write('\nSTART ' + now() + '\n' + json.dumps(argv) + '\n')
        stream.flush()
        process = subprocess.Popen(argv, cwd=cwd or ROOT, stdout=stream, stderr=subprocess.STDOUT, env=os.environ)
        status('running', label, argv=argv, child_pid=process.pid, log=str(log))
        code = process.wait()
    receipt = {'at': now(), 'argv': argv, 'exit_code': code, 'expected_exit_code': expected,
               'seconds': time.monotonic() - started, 'log': str(log)}
    dump(ROOT / 'artifacts' / (label + '-receipt.json'), receipt)
    record(ROOT / 'artifacts/execution-receipts.jsonl', {'label': label, **receipt})
    if code != expected:
        raise RuntimeError(f'{label} exit {code}, expected {expected}; see {log}')


def training_command(stage, smoke=False, resume=False, stop_after=None):
    python = CONFIG['python']
    if smoke:
        argv = [python]
        if stage == 'ddp_smoke':
            argv += ['-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc_per_node=' + str(len(CONFIG['gpus']))]
        argv += [str(ROOT / 'training/run_stage.py'), '--stage', stage, '--script', 'train_sft_vlm.py']
        if stop_after: argv += ['--stop-after', str(stop_after)]
        options = {'save_weight': stage, 'from_weight': 'llm', 'epochs': 2, 'batch_size': 2,
                   'learning_rate': 5e-6, 'freeze_llm': 1, 'max_seq_len': 768,
                   'data_path': str(ROOT / 'data/smoke.parquet'), **CONFIG['common'],
                   'log_interval': 1, 'save_interval': 1 if stage == 'ddp_smoke' else 2}
    else:
        cfg = next(x for x in CONFIG['stages'] if x['name'] == stage)
        argv = [python, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                '--nproc_per_node=' + str(len(CONFIG['gpus'])), str(ROOT / 'training/run_stage.py'),
                '--stage', stage, '--script', cfg['script']]
        options = {key: value for key, value in cfg.items() if key not in ['name', 'script', 'data']}
        options.update(CONFIG['common'])
        options['data_path'] = str(ROOT / 'data' / cfg['data'])
    options.update(CONFIG['architecture'])
    options.update(device='cuda:0', from_resume=int(resume))
    for key, value in options.items(): argv += ['--' + key, str(value)]
    return argv


def complete_report():
    manifest = json.loads((ROOT / 'artifacts/data_manifest.json').read_text())
    records = {}
    for cfg in CONFIG['stages']:
        name = cfg['name']
        folder = ROOT / 'runs' / name
        audit = json.loads((folder / 'checkpoint_audit.json').read_text())
        receipts = [json.loads((folder / f'receipt-success-rank{rank}.json').read_text()) for rank in range(len(CONFIG['gpus']))]
        attempts = [[json.loads(line) for line in (folder / f'attempt-receipts-rank{rank}.jsonl').read_text().splitlines()]
                    for rank in range(len(CONFIG['gpus']))]
        dataset = manifest['datasets'][name]
        seconds = max(sum(x['wall_seconds'] for x in per_rank) for per_rank in attempts)
        resumed = any(x['event'] == 'resume_loaded' for x in [json.loads(line) for line in (folder / 'events.jsonl').read_text().splitlines()])
        padded_samples = math.ceil(dataset['rows'] / len(CONFIG['gpus'])) * len(CONFIG['gpus']) * cfg['epochs']
        records[name] = {'dataset_rows': dataset['rows'], 'epochs': cfg['epochs'], 'nominal_completed_padded_sample_exposures': padded_samples,
            'adam_updates': audit['adam_step_max'], 'trainable_parameters': audit['trainable_parameters'],
            'global_batch_size': cfg['batch_size'] * len(CONFIG['gpus']), 'observed_process_wall_seconds': seconds,
            'timing_scope': 'Maximum per-rank sum of recorded invocation durations; excludes controller gaps and does not count reprocessed samples.',
            'resumed': resumed, 'padded_samples_per_process_wall_second': None if resumed else padded_samples / seconds,
            'rank_peak_allocated_bytes': [x['peak_allocated_bytes'] for x in receipts],
            'checkpoint_audit': audit, 'export_verification': json.loads((ROOT / 'exports' / name / 'export_verification.json').read_text())}
    dump(ROOT / 'results.json', {'at': now(), 'status': 'complete', 'config': CONFIG, 'stages': records,
        'evaluation_scope': CONFIG['evaluation'], 'capability_improvement_verified': False})
    lines = ['# MiniMind-V 官方两阶段全链路复现', '', '全部训练、检查点审计、导出校验和网站示例评估均已完成。', '',
             '| 阶段 | 官方样本行数 | 轮次 | 全局 batch | Adam 更新 | 进程耗时（小时） |', '|---|---:|---:|---:|---:|---:|']
    for name, value in records.items():
        lines.append(f"| {name} | {value['dataset_rows']:,} | {value['epochs']} | {value['global_batch_size']} | {value['adam_updates']:,} | {value['observed_process_wall_seconds']/3600:.2f} |")
    lines += ['', '## 可用于简历的工程事实', '',
        f"基于 MiniMind-V 复现冻结 SigLIP 视觉编码器、MLP 投影与 65M 语言骨干的两阶段多模态训练；完成 {records['pretrain']['dataset_rows']:,} 条图文对齐样本和 {records['sft']['dataset_rows']:,} 条视觉指令样本各 2 轮训练，使用 4 张 RTX 4090 的 DDP/BF16，验证原生 Adam 恢复、冻结参数一致性及 PyTorch→Transformers 导出的一致性，并部署本地图片问答 WebUI。", '',
        '## 证据与边界', '',
        '- 使用官方发布的语言基座与视觉编码器；“从零”仅指随机初始化的视觉投影与跨模态对齐，不是从零预训练语言/视觉基础模型。',
        '- 使用固定版本的完整官方 Parquet 数据；未使用旧实验的自训语言基座、过滤子集或 Qwen 适配结果。',
        '- 原仓库支持 DDP；本次沿用每卡 batch，4 卡会改变全局 batch 和更新次数，不能称逐步等价于单卡训练。',
        '- 最小训练补丁仅包含 DDP 保存同步、原子写入、日志跨卡均值与非有限损失检查，完整 diff 保存在 artifacts/trainer.patch。',
        '- 断点烟测证明模型/Adam/位置恢复可继续训练；原生增强 RNG 不完整保存，不宣称逐比特恢复。',
        '- 网站提供的六张示例属于定性推理演示，不是留出测试集；没有 accuracy、OCR 增益或优于官方的结论。',
        '- 参数、更新次数、吞吐和显存口径见 results.json；观察到的进程耗时包含初始化/日志/保存，非独占硬件基准。',
        '- 导出校验以原生 FP16 发布文件为参考，包含全持久张量与一张图片前向；不宣称与 FP32 训练状态无损等价。', '',
        '阶段日志在 logs/，逐次 loss 在 runs/*/events.jsonl，检查点审计在 runs/*/checkpoint_audit.json，原始预测在 evaluation/。']
    (ROOT / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    # Keep the plot derived solely from logged train loss; never label it validation accuracy.
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for axis, cfg in zip(axes, CONFIG['stages']):
            rows = [json.loads(line) for line in (ROOT / 'runs' / cfg['name'] / 'events.jsonl').read_text().splitlines()]
            rows = [x for x in rows if 'loss' in x]
            axis.plot([(x['epoch']-1)*x['steps']+x['step'] for x in rows], [x['loss'] for x in rows], linewidth=.8)
            axis.set(title=cfg['name'], xlabel='Optimizer update', ylabel='Logged training loss (rank mean)')
            axis.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(ROOT / 'training_loss.png', dpi=180)
        plt.close(fig)
    except ImportError:
        (ROOT / 'artifacts/plot_unavailable.txt').write_text('matplotlib unavailable; all original loss points retained in JSONL.\n')


if __name__ == '__main__':
    lock = (ROOT / '.pipeline.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ.update(CUDA_VISIBLE_DEVICES=','.join(map(str, CONFIG['gpus'])), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
        HF_DATASETS_OFFLINE='1', HF_HOME=str(ROOT / 'cache/huggingface'), TMPDIR=str(ROOT / 'tmp'),
        PYTHONPATH=str(ROOT / 'training') + ':' + str(ROOT / 'src'))
    lease_handles = []
    try:
        lease_dir = Path.home() / '.gpu_leases'
        lease_dir.mkdir(exist_ok=True)
        for index in CONFIG['gpus']:
            handle = (lease_dir / f'gpu-{index}.lock').open('a+')
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            handle.seek(0); handle.truncate(); handle.write(json.dumps({'pid': os.getpid(), 'project': str(ROOT), 'at': now()})); handle.flush()
            lease_handles.append(handle)
        gpu_check()
        python, verifier = CONFIG['python'], str(ROOT / 'training/verify_and_evaluate.py')
        if not (ROOT / 'runs/smoke/checkpoint_audit.json').exists():
            smoke_checkpoint = ROOT / 'runs/smoke/checkpoints/smoke_768_resume.pth'
            if not smoke_checkpoint.exists(): execute('smoke-stop', training_command('smoke', smoke=True, stop_after=2), expected=75)
            execute('smoke-resume', training_command('smoke', smoke=True, resume=True))
            execute('smoke-audit', [python, verifier, '--stage', 'smoke'])
        if not (ROOT / 'exports/smoke/export_verification.json').exists():
            execute('smoke-export', [python, verifier, '--stage', 'smoke', '--export'])
        if not (ROOT / 'runs/ddp_smoke/checkpoint_audit.json').exists():
            gpu_check()
            execute('ddp-smoke', training_command('ddp_smoke', smoke=True))
            execute('ddp-smoke-audit', [python, verifier, '--stage', 'ddp_smoke'])
        if not (ROOT / 'evaluation/initial/summary.json').exists():
            execute('evaluate-initial', [python, verifier, '--evaluate', '--label', 'initial', '--weight', str(ROOT / 'weights/llm_768.pth')])
        if not (ROOT / 'evaluation/official-release/summary.json').exists():
            execute('evaluate-official-release', [python, verifier, '--evaluate', '--label', 'official-release', '--weight', str(ROOT / 'weights/official_sft_vlm_768.pth')])
        for cfg in CONFIG['stages']:
            name = cfg['name']
            audited = ROOT / 'runs' / name / 'checkpoint_audit.json'
            if not audited.exists():
                gpu_check()
                success = all((ROOT / 'runs' / name / f'receipt-success-rank{rank}.json').exists() for rank in range(len(CONFIG['gpus'])))
                if not success:
                    resume = (ROOT / 'runs' / name / 'checkpoints' / f"{cfg['save_weight']}_768_resume.pth").exists()
                    execute('train-' + name, training_command(name, resume=resume))
                execute('verify-export-' + name, [python, verifier, '--stage', name])
            elif not (ROOT / 'exports' / name / 'export_verification.json').exists():
                execute('verify-export-' + name, [python, verifier, '--stage', name])
            if not (ROOT / 'evaluation' / name / 'summary.json').exists():
                execute('evaluate-' + name, [python, verifier, '--evaluate', '--label', name,
                    '--weight', str(ROOT / 'runs' / name / 'out' / f"{cfg['save_weight']}_768.pth")])
        if not (ROOT / 'artifacts/upstream-cli-receipt.json').exists():
            execute('upstream-cli', [python, str(ROOT / 'src/eval_vlm.py'), '--load_from', 'model',
                '--save_dir', str(ROOT / 'runs/sft/out'), '--weight', 'sft_vlm', '--device', 'cuda:0', '--max_new_tokens', '128'], cwd=ROOT / 'src')
        complete_report()
        status('complete', 'all_training_evaluation_export_complete', report=str(ROOT / 'REPORT.md'))
    except BaseException as error:
        status('failed', 'manual_attention_required', error=repr(error), traceback=traceback.format_exc())
        raise
