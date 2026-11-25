#!/usr/bin/env bash
# One-time setup: Python virtualenv + dependencies (+ optional frontend rebuild). Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
"$PY" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,13), f"Python 3.10-3.13 required, found {sys.version.split()[0]}"'
[ -d .venv ] || "$PY" -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
[ -f .env ] || cp .env.example .env
if command -v npm >/dev/null 2>&1 && [ "${REBUILD_FRONTEND:-0}" = "1" ]; then
  (cd frontend && npm ci --no-audit --no-fund 2>/dev/null || npm install --no-audit --no-fund; npm run build)
fi
echo
echo "Setup finished. Start the app with ./run.sh   (optional: edit .env to add GROQ_API_KEY)"
