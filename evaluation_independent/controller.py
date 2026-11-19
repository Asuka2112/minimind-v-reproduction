"""Durable one-shot controller; no scheduled polling or automated follow-up."""
import fcntl
import subprocess
import time
import traceback
from common import *

def main():
    (HERE / 'artifacts').mkdir(exist_ok=True)
    lock = (HERE / 'artifacts/controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (HERE / 'summary.json').exists():
        print('Already complete; aggregate.py can independently recompute scores.'); return
    leases = []; children = []; logs = []
    state = {'status': 'starting', 'at': now(), 'pid': os.getpid(), 'protocol_sha256': sha(HERE / 'protocol.json')}
    try:
        for gpu in range(4):
            path = Path.home() / f'.gpu_leases/gpu-{gpu}.lock'
            path.parent.mkdir(exist_ok=True)
            lease = path.open('a'); fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB); leases.append(lease)
        inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,uuid', '--format=csv,noheader,nounits'], text=True)
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
        occupied = {line.split(',')[0].strip() for line in processes.splitlines() if line.strip()}
        for line in inventory.splitlines():
            index, memory, uuid = [x.strip() for x in line.split(',')]
            if int(index) in range(4): assert int(memory) <= 256 and uuid not in occupied, f'GPU occupied: {line}'
        dump(HERE / 'artifacts/gpu_start.json', {'at': now(), 'inventory': inventory, 'compute_processes': processes})
        for name, command, gpu in [('overlap', ['audit_overlap.py'], None)] + [(m, ['evaluate.py', '--model', m], i) for i, m in enumerate(MODELS)]:
            env = os.environ.copy(); env.update(TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2', TMPDIR=str(TRAIN / 'tmp'), HF_HOME=str(TRAIN / 'cache/hf'), HF_HUB_OFFLINE='1')
            if gpu is not None: env['CUDA_VISIBLE_DEVICES'] = str(gpu)
            log = (HERE / f'artifacts/{name}.log').open('a'); logs.append(log)
            child = subprocess.Popen([PYTHON] + command, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT)
            children.append((name, child))
        state.update(status='running', started_at=now(), workers={n: p.pid for n, p in children}); dump(HERE / 'status.json', state)
        failed = False
        while any(p.poll() is None for _, p in children):
            exits = {n: p.poll() for n, p in children}
            state.update(at=now(), exits=exits); dump(HERE / 'status.json', state)
            if any(v is not None and v != 0 for v in exits.values()): failed = True; break
            time.sleep(5)
        if failed:
            for _, p in children:
                if p.poll() is None: p.terminate()
            for _, p in children: p.wait()
        exits = {n: p.wait() for n, p in children}
        assert all(v == 0 for v in exits.values()), exits
        subprocess.run([PYTHON, 'aggregate.py'], cwd=HERE, check=True)
        state.update(status='complete', completed_at=now(), exits=exits, summary_sha256=sha(HERE / 'summary.json'))
    except Exception:
        state.update(status='failed', failed_at=now(), traceback=traceback.format_exc())
        raise
    finally:
        dump(HERE / 'status.json', state)
        for log in logs: log.close()

if __name__ == '__main__': main()
