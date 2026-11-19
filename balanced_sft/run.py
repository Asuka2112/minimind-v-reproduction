"""Run the frozen protocol as a durable task named 评测."""
import argparse
import ctypes
import fcntl
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import traceback

from common import BASE, BASE_SHA, HERE, PRIOR, SOURCE, dump, load, now, sha

MODELS = ('base', 'control', 'targeted')

def set_name():
    if sys.platform=='linux':
        ctypes.CDLL(None).prctl(15,ctypes.c_char_p('评测'.encode()),0,0,0)

def acquire(limit):
    info=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,uuid','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    used={line.split(',')[0].strip() for line in apps.splitlines()}
    folder=Path.home()/'.gpu_leases'
    folder.mkdir(exist_ok=True)
    devices={}
    for row in info.splitlines():
        index,mem,uuid=[cell.strip() for cell in row.split(',')]
        if int(mem)>256 or uuid in used:
            continue
        lease=(folder/f'gpu-{index}.lock').open('a')
        try:
            fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            lease.close()
            continue
        devices[index]=lease
        if len(devices)==limit:
            break
    return devices

def validate_protocol():
    data=load(HERE/'protocol.json')
    source=SOURCE/'model'
    for name,digest in data['code_sha256'].items():
        assert sha(HERE/name)==digest,name
    for name,digest in data['upstream_file_sha256'].items():
        assert sha(source/name)==digest,name
    for name,digest in data['prior_source_sha256'].items():
        assert sha(PRIOR/name)==digest,name
    for name,digest in data['quality_sha256'].items():
        assert sha(HERE/name)==digest,name
    assert sha(HERE/'data/manifest.json')==data['data_manifest_sha256']
    assert sha(BASE)==BASE_SHA
    return sha(HERE/'protocol.json')

def check_receipt(rel, protocol_hash):
    file=HERE/rel
    if not file.exists():
        return False
    receipt=load(file)
    assert receipt['protocol_sha256']==protocol_hash, str(file)
    if 'split' in receipt:
        assert sha(HERE/'predictions'/f'{file.stem}.jsonl')==receipt['prediction_sha256']
        if receipt['split']=='test':
            model,seed=receipt['model'],receipt['seed']
            assert sha(HERE/'predictions'/f'text-{model}-{seed}.jsonl')==receipt['text_sha256']
    elif receipt['arm'] in ('control','targeted'):
        assert sha(HERE/'weights'/f'{file.stem}.pth')==receipt['checkpoint_sha256']
    return True

def controller():
    with (HERE/'run.lock').open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        status=HERE/'status.json'
        if status.exists() and load(status)['status']=='complete':
            print('评测已经完成。')
            return
        state={'name':'评测','pid':os.getpid(),'status':'starting','started_at':now(),'completed_stages':[]}
        leases={}
        children={}
        logs=HERE/'logs'
        logs.mkdir(exist_ok=True)
        (HERE/'tmp').mkdir(exist_ok=True)
        try:
            protocol_hash=validate_protocol()
            state.update(protocol_sha256=protocol_hash)
            while not leases:
                leases=acquire(2)
                if leases:
                    break
                if state['status']!='queued':
                    print('评测等待空闲 GPU。',flush=True)
                state.update(status='queued',stage='waiting_for_idle_gpu',at=now())
                dump(status,state)
                time.sleep(45)
            state.update(status='running',gpus=list(leases))
            dump(status,state)

            def stage(label,jobs):
                stage_specs=[j for j in jobs if not check_receipt(j[2],protocol_hash)]
                state.update(stage=label,at=now(),workers={})
                dump(status,state)
                while stage_specs or children:
                    free=[g for g in leases if g not in children]
                    for gpu in free:
                        if not stage_specs:
                            break
                        name,args,_=stage_specs.pop(0)
                        env=os.environ.copy()
                        env.update(CUDA_VISIBLE_DEVICES=gpu,HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
                                   OMP_NUM_THREADS='2',TMPDIR=str(HERE/'tmp'))
                        with (logs/f'{name}.log').open('a') as log:
                            command=['../runtime/bin/python','-u','run.py','--worker',*args]
                            p=subprocess.Popen(command,cwd=HERE,env=env,stdout=log,stderr=subprocess.STDOUT)
                        children[gpu]=(name,p)
                    state.update(at=now(),workers={n:{'pid':p.pid,'gpu':g} for g,(n,p) in children.items()})
                    dump(status,state)
                    for gpu,(name,p) in list(children.items()):
                        rc=p.poll()
                        if rc is not None:
                            if rc!=0:
                                raise RuntimeError(f'{name} 退出码 {rc}。详见 logs/{name}.log')
                            del children[gpu]
                    if stage_specs or children:
                        time.sleep(3)
                state['completed_stages'].append(label)
                assert all(check_receipt(j[2],protocol_hash) for j in jobs)
                dump(status,state)

            def train(seed):
                stage(f'train-{seed}',[(f'{m}-{seed}',['train.py',m,'--seed',str(seed)],f'receipts/{m}-{seed}.json')
                                       for m in ('control','targeted')])

            def evaluate(split,seed,models):
                stage(f'{split}-{seed}',[(f'{split}-{m}-{seed}',['evaluate.py',m,split,'--seed',str(seed)],
                                         f'receipts/{split}-{m}-{seed}.json') for m in models])
                from metrics import summarize
                full_models=MODELS if seed!=42 else models
                return summarize(split,seed,full_models)

            stage('smoke-32',[('smoke',['train.py','smoke'],'receipts/smoke.json')])
            assert load(HERE/'receipts/smoke.json')['pass_gate']
            dev=evaluate('dev',42,('base',))
            policy=load(HERE/'protocol.json')['eval']
            scores={cap:dev['models']['base'][str(cap)]['conditions']['original']['semantic'] for cap in policy['caps']}
            selection={'selected_cap':max(scores,key=lambda cap:(scores[cap],-cap)), 'dev_semantic':scores,
                       'protocol_sha256':protocol_hash, 'rule':policy['selection']}
            choice=HERE/'decode_selection.json'
            if choice.exists():
                selection['dev_semantic']={str(k):v for k,v in scores.items()}
                assert selection==load(choice)
            else:
                dump(choice,selection)
            train(42)
            result=evaluate('test',42,MODELS)
            evaluate('challenge',42,MODELS)
            state.update(first_seed_gates=result['gates'],first_seed_passed=result['all_gates'])
            if result['all_gates']:
                train(123)
                result=evaluate('test',123,('control','targeted'))
                evaluate('challenge',123,('control','targeted'))
                state.update(second_seed_gates=result['gates'],second_seed_passed=result['all_gates'])
            state.update(status='complete',stage='awaiting_semantic_review',workers={},completed_at=now())
            dump(status,state)
        except BaseException:
            for _,p in children.values():
                if p.poll() is None:
                    p.terminate()
            for _,p in children.values():
                p.wait()
            state.update(status='failed',at=now(),error=traceback.format_exc())
            dump(status,state)
            raise
        finally:
            for lock in leases.values():
                lock.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    set_name()
    if args.worker:
        entry,*argv=args.worker
        assert entry in ('train.py','evaluate.py')
        sys.argv=[entry,*argv]
        runpy.run_path(str(HERE/entry),run_name='__main__')
    else:
        controller()
