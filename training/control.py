"""Manual status and result download from the user's saved SSH host."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

LOCAL = Path(__file__).resolve().parent
CONFIG = json.loads((LOCAL / 'config.json').read_text())


def status():
    code = '''import json,os,pathlib,sys
r=pathlib.Path(sys.argv[1]);p=r/'pipeline_status.json'
s=json.loads(p.read_text()) if p.exists() else {'status':'not_started'}
try:
 pid=int(s.get('pid',0));assert pid>0
 os.kill(pid,0);s['process_alive']=True
except AssertionError:s['process_alive']=False
except (OSError,ValueError):s['process_alive']=False
for stage in ['pretrain','sft']:
 p=r/'runs'/stage/'progress.json'
 if p.exists():s[stage+'_progress']=json.loads(p.read_text())
log=pathlib.Path(s.get('log',str(r/'logs/pipeline.log')))
if log.is_file():s['log_tail']=log.read_text(errors='replace').splitlines()[-12:]
print(json.dumps(s,ensure_ascii=False,indent=2))
'''
    command = shlex.join([CONFIG['python'], '-c', code, CONFIG['remote_root']])
    result = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', CONFIG['ssh_host'], command], text=True)
    parsed = json.loads(result)
    (LOCAL / 'server_status.json').write_text(json.dumps(parsed, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(parsed, ensure_ascii=False, indent=2))


def sync():
    destination = LOCAL / 'results'
    destination.mkdir(exist_ok=True)
    excludes = ['data/***', 'weights/***', 'cache/***', 'tmp/***', 'upstream/***', 'src/***',
                'runs/*/checkpoints/***', 'runs/*/out/***', 'runs/*/model', 'runs/*/trainer/***', '.pipeline.lock', '*.tar']
    argv = ['rsync', '-a', '--partial']
    for pattern in excludes: argv.append('--exclude=' + pattern)
    argv += [CONFIG['ssh_host'] + ':' + CONFIG['remote_root'] + '/', str(destination) + '/']
    subprocess.run(argv, check=True)
    status()
    print('已同步到：' + str(destination))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='手动检查全链路训练与下载结果')
    parser.add_argument('action', choices=['status', 'sync'])
    args = parser.parse_args()
    status() if args.action == 'status' else sync()
