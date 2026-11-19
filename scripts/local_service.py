"""Only stop the local_app.py process identified by this checkout's PID file."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
PID_FILE = ROOT / '.local-app.pid'


def service():
    data = json.loads(PID_FILE.read_text())
    pid = int(data['pid'])
    command = subprocess.check_output(['ps', '-p', str(pid), '-o', 'command='], text=True)
    if str(ROOT / 'local_app.py') not in command:
        raise RuntimeError('PID 已过期，未操作其他进程。')
    return pid, data['url']


if __name__ == '__main__':
    try:
        pid, url = service()
        if sys.argv[1] == 'stop':
            os.kill(pid, signal.SIGTERM)
            PID_FILE.unlink(missing_ok=True)
            print('已停止 MiniMind-V 本地服务。')
        else:
            with urlopen(url + '/config', timeout=2) as response:
                config = json.load(response)
            if config.get('title') != 'MiniMind-V · 本地视觉问答':
                raise RuntimeError('端口上不是 MiniMind-V 服务。')
            print(url)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f'本项目服务未运行：{error}')
        sys.exit(1)
