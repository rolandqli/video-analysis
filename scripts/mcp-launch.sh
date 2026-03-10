#!/usr/bin/env bash
# Launch an MCP server with paths derived from this script's location.
# Usage: mcp-launch.sh <server-name>
# Server names: video-analysis | sam2-segmentation
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SERVER="$1"
case "$SERVER" in
  video-analysis)
    exec "$PROJECT_ROOT/.venv-video-analysis/bin/python" "$PROJECT_ROOT/server.py"
    ;;
  sam2-segmentation)
    exec "$PROJECT_ROOT/.venv-sam2-segmentation/bin/python" "$PROJECT_ROOT/sam_server.py"
    ;;
  *)
    echo "Unknown server: $SERVER (use: video-analysis | sam2-segmentation)" >&2
    exit 1
    ;;
esac
