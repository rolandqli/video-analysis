#!/usr/bin/env bash
# Launch an MCP server with paths derived from this script's location.
# Usage: mcp-launch.sh <server-name>
# Server names: video-analysis | sam2-segmentation | resources | orchestrator
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SERVER="$1"
case "$SERVER" in
  video-analysis)
    exec "$PROJECT_ROOT/.venv-video-analysis/bin/python" "$PROJECT_ROOT/servers/video_analysis.py"
    ;;
  sam2-segmentation)
    exec "$PROJECT_ROOT/.venv-sam2-segmentation/bin/python" "$PROJECT_ROOT/servers/sam2_segmentation.py"
    ;;
  resources)
    exec "$PROJECT_ROOT/.venv-resources/bin/python" "$PROJECT_ROOT/servers/resources.py"
    ;;
  orchestrator)
    exec "$PROJECT_ROOT/.venv-orchestrator/bin/python" "$PROJECT_ROOT/servers/orchestrator.py"
    ;;
  *)
    echo "Unknown server: $SERVER (use: video-analysis | sam2-segmentation | resources | orchestrator)" >&2
    exit 1
    ;;
esac
