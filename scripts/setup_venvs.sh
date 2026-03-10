#!/usr/bin/env bash
# Create per-service virtual environments with isolated dependencies.
# Requires Python 3.10+ (mcp package). Use PYTHON=... to override interpreter.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
for p in python3.12 python3.11 python3.10 python3; do
  if command -v "$p" &>/dev/null && "$p" -c 'import sys; exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
    PYTHON="$p"
    break
  fi
done

create_venv() {
  local name=$1
  local req=$2
  local venv=".venv-${name}"
  [[ -d "$venv" ]] && rm -rf "$venv"
  echo "Creating $venv..."
  "$PYTHON" -m venv "$venv"
  "$venv/bin/pip" install --upgrade pip
  "$venv/bin/pip" install -r "$req"
  echo "  -> $venv ready"
}

create_venv "video-analysis" "requirements/video-analysis.txt"
create_venv "sam2-segmentation" "requirements/sam2-segmentation.txt"
create_venv "resources" "requirements/resources.txt"
create_venv "orchestrator" "requirements/orchestrator.txt"

echo ""
echo "On Mac without CUDA, if SAM 2 install failed, run:"
echo "  SAM2_BUILD_CUDA=0 .venv-sam2-segmentation/bin/pip install 'git+https://github.com/facebookresearch/sam2.git'"
echo ""

