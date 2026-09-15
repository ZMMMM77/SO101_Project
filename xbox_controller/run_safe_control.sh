#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export HF_HOME="$PROJECT_DIR/.hf-home"
export HF_LEROBOT_HOME="$PROJECT_DIR/.lerobot"
export PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS=1

cd "$SCRIPT_DIR"
source "$PROJECT_DIR/.venv/bin/activate"

exec python -u xbox_official_control.py control \
  --execute \
  --port /dev/cu.usbmodem5B3E0888091 \
  --robot-id my_follower \
  --hz 20 \
  --linear-speed-mm-s 15 \
  --max-cartesian-step-mm 1.5 \
  --gripper-speed-deg-s 20 \
  --wrist-roll-speed-deg-s 35 \
  --deploy-speed-deg-s 30 \
  --max-relative-target-deg 8 \
  "$@"
