#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  bundled="$HOME/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
  runtime="${PYTHON_BIN:-python3}"
  if ! "$runtime" -c 'import sys; assert sys.version_info[:2] == (3, 12)' 2>/dev/null; then
    runtime="$bundled"
  fi
  if ! "$runtime" -c 'import sys; assert sys.version_info[:2] == (3, 12)' 2>/dev/null; then
    echo '需要 Python 3.12。请通过 PYTHON_BIN=/路径/python3.12 bash setup_local.sh 指定解释器。' >&2
    exit 1
  fi
  "$runtime" -m venv .venv
fi
.venv/bin/python -m pip install -r requirements-local.lock.txt
.venv/bin/python scripts/download_local_models.py
echo '安装完成。双击 一键启动.command 或运行 .venv/bin/python local_app.py --open'
