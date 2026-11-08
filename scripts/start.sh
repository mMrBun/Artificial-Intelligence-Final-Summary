#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -f .env ]]; then
  cp .env.example .env
  chmod 600 .env
fi
export PYTHONUNBUFFERED=1
export GRADIO_ANALYTICS_ENABLED=false
export HF_HUB_DISABLE_TELEMETRY=1
exec .venv/bin/python -m ragkb.main

