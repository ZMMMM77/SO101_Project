# Source this file from zsh before running SO-101 commands:
#   source ./activate_so101.zsh

SO101_PROJECT_ROOT="${${(%):-%N}:A:h}"
export SO101_PROJECT_ROOT
export HF_HOME="$SO101_PROJECT_ROOT/.hf-home"
export HF_LEROBOT_HOME="$SO101_PROJECT_ROOT/.lerobot"
export UV_CACHE_DIR="$SO101_PROJECT_ROOT/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$SO101_PROJECT_ROOT/.uv-python"

source "$SO101_PROJECT_ROOT/.venv/bin/activate"
cd "$SO101_PROJECT_ROOT"

echo "SO-101 environment active: $VIRTUAL_ENV"
echo "LeRobot data directory:    $HF_LEROBOT_HOME"
