"""All MCP resources (asset listings) in one place."""

import json
import os

_ASSETS = os.path.join(os.getcwd(), "assets")


def list_videos() -> str:
    """List videos in assets/videos."""
    videos_dir = os.path.join(_ASSETS, "videos")
    if not os.path.isdir(videos_dir):
        return json.dumps({"videos": [], "path": videos_dir, "count": 0})
    entries = sorted(os.listdir(videos_dir))
    videos = [
        e for e in entries
        if e.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm"))
    ]
    return json.dumps({"videos": videos, "path": videos_dir, "count": len(videos)})


def list_frames() -> str:
    """List frame folders in assets/frames (output of video_to_frames)."""
    frames_dir = os.path.join(_ASSETS, "frames")
    if not os.path.isdir(frames_dir):
        return json.dumps({"folders": [], "path": frames_dir, "count": 0})
    folders = sorted(
        d for d in os.listdir(frames_dir)
        if os.path.isdir(os.path.join(frames_dir, d))
    )
    return json.dumps({"folders": folders, "path": frames_dir, "count": len(folders)})


def list_masks() -> str:
    """List mask outputs in assets/masks (one folder per segmented input)."""
    masks_dir = os.path.join(_ASSETS, "masks")
    if not os.path.isdir(masks_dir):
        return json.dumps({"inputs": [], "path": masks_dir, "count": 0})
    inputs = sorted(
        d for d in os.listdir(masks_dir)
        if os.path.isdir(os.path.join(masks_dir, d))
    )
    return json.dumps({"inputs": inputs, "path": masks_dir, "count": len(inputs)})
