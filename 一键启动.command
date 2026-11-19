#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  bash setup_local.sh
fi
if .venv/bin/python scripts/local_service.py status >/dev/null 2>&1; then
  open http://127.0.0.1:8888
  exit 0
fi
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false GRADIO_ANALYTICS_ENABLED=False
exec .venv/bin/python "$PWD/local_app.py" --open
