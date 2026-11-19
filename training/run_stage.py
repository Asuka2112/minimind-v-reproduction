"""Isolated entry point around the minimally patched upstream trainer."""
import argparse
import os
from pathlib import Path
import runpy
import sys
import time

from instrument import ROOT, dump, install, now, record
import torch
import torch.distributed as dist


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True)
    parser.add_argument('--script', required=True)
    parser.add_argument('--stop-after', type=int)
    args, trainer_args = parser.parse_known_args()
    folder = ROOT / 'runs' / args.stage
    (folder / 'trainer').mkdir(parents=True, exist_ok=True)
    os.chdir(folder / 'trainer')
    os.environ.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    started = install(args.stage, args.stop_after)
    exit_code = 0
    try:
        sys.argv = [args.script] + trainer_args
        runpy.run_path(str(ROOT / 'src/trainer' / args.script), run_name='__main__')
    except SystemExit as error:
        exit_code = error.code if isinstance(error.code, int) else 1
        raise
    except BaseException:
        exit_code = 1
        raise
    finally:
        rank = int(os.environ.get('RANK', '0'))
        receipt = {'at': now(), 'exit_code': exit_code, 'rank': rank, 'world_size': int(os.environ.get('WORLD_SIZE', '1')),
                   'wall_seconds': time.monotonic() - started, 'argv': sys.argv,
                   'peak_allocated_bytes': torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
                   'peak_reserved_bytes': torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None}
        label = 'stopped' if exit_code == 75 else 'success' if exit_code == 0 else 'failed'
        dump(folder / f'receipt-{label}-rank{rank}.json', receipt)
        record(folder / f'attempt-receipts-rank{rank}.jsonl', receipt)
        if dist.is_initialized():
            dist.destroy_process_group()
