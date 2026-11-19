#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
.venv/bin/python scripts/local_service.py stop
