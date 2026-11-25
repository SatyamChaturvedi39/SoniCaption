#!/usr/bin/env bash
# Starts SonicCaption (API + web UI on one port). Open http://localhost:8000
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] || { echo "Run ./setup.sh first"; exit 1; }
. .venv/bin/activate
PORT="${PORT:-8000}"
echo "SonicCaption -> http://localhost:${PORT}"
exec python -m uvicorn backend.app.main:app --host 0.0.0.0 --port "$PORT"
