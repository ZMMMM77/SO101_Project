#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export HF_HOME="$PROJECT_DIR/.hf-home"
export HF_LEROBOT_HOME="$PROJECT_DIR/.lerobot"

cd "$SCRIPT_DIR"
source "$PROJECT_DIR/.venv/bin/activate"
exec python xbox_so101_control.py input-test
