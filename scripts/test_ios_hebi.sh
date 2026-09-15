#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$ROOT_DIR/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

export HF_HOME="$PROJECT_DIR/.hf-home"
export HF_LEROBOT_HOME="$PROJECT_DIR/.lerobot"
cd "$ROOT_DIR"
source "$VENV_DIR/bin/activate"

python - <<'PY'
import time
import hebi

print("Scanning for HEBI Mobile I/O for 10 seconds...")
print("Expected family/name: HEBI / mobileIO")
print("Keep the iPhone app open and on the same Wi-Fi as this Mac.")

lookup = hebi.Lookup()
time.sleep(10)

print("Raw entry list:")
print(lookup.entrylist)

group = lookup.get_group_from_names(["HEBI"], ["mobileIO"])
if group is None:
    raise SystemExit(
        "Mobile I/O not found. Check iOS Local Network permission, Wi-Fi, and app family/name settings."
    )

print(f"Mobile I/O found: group size={group.size}")
PY
