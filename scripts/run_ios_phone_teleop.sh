#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$ROOT_DIR/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
DEFAULT_PORT="/dev/cu.usbmodem5B3E0888091"

export HF_HOME="$PROJECT_DIR/.hf-home"
export HF_LEROBOT_HOME="$PROJECT_DIR/.lerobot"

if [ "${1:-}" != "" ] && [[ "${1:-}" != --* ]]; then
  ROBOT_PORT="$1"
  shift
else
  ROBOT_PORT="$DEFAULT_PORT"
fi

cd "$ROOT_DIR"
source "$VENV_DIR/bin/activate"

python "$ROOT_DIR/ios_phone_so101_joint_teleop.py" --port "$ROBOT_PORT" "$@"
