#!/usr/bin/env bash
# Create per-service virtual environments with uv (https://docs.astral.sh/uv/).
# Requires: uv (brew install uv), Python 3.10+
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v uv &>/dev/null; then
  echo "uv not found. Install: brew install uv" >&2
  exit 1
fi

create_venv() {
  local name=$1
  local req=$2
  local venv=".venv-${name}"
  [[ -d "$venv" ]] && rm -rf "$venv"
  echo "Creating $venv..."
  uv venv "$venv" --python 3.10
  uv pip install -r "$req" --python "$venv/bin/python"
  echo "  -> $venv ready"
}

create_venv "video-analysis" "requirements/video-analysis.txt"
create_venv "sam2-segmentation" "requirements/sam2-segmentation.txt"
create_venv "resources" "requirements/resources.txt"
create_venv "orchestrator" "requirements/orchestrator.txt"
create_venv "test" "requirements/test.txt"

echo ""
echo "On Mac without CUDA, if SAM 2 install failed, run:"
echo "  SAM2_BUILD_CUDA=0 uv pip install --python .venv-sam2-segmentation/bin/python 'git+https://github.com/facebookresearch/sam2.git'"
echo ""

