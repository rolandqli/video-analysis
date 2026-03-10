#!/usr/bin/env bash
# Launch an MCP server with paths derived from this script's location.
# Uses python -m so project root is on sys.path (no sys.path hacks).
# Usage: mcp-launch.sh <server-name>
# Server names: video-analysis | sam2-segmentation | resources | orchestrator
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SERVER="$1"
case "$SERVER" in
  video-analysis)
    cd "$PROJECT_ROOT" && exec "$PROJECT_ROOT/.venv-video-analysis/bin/python" -m servers.video_analysis
    ;;
  sam2-segmentation)
    cd "$PROJECT_ROOT" && exec "$PROJECT_ROOT/.venv-sam2-segmentation/bin/python" -m servers.sam2_segmentation
    ;;
  resources)
    cd "$PROJECT_ROOT" && exec "$PROJECT_ROOT/.venv-resources/bin/python" -m servers.resources
    ;;
  orchestrator)
    cd "$PROJECT_ROOT" && exec "$PROJECT_ROOT/.venv-orchestrator/bin/python" -m servers.orchestrator
    ;;
  *)
    echo "Unknown server: $SERVER (use: video-analysis | sam2-segmentation | resources | orchestrator)" >&2
    exit 1
    ;;
esac
