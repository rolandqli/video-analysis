"""Unit tests for resources (list_videos, list_frames, list_masks)."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from resources.resources import list_frames, list_masks, list_videos


def test_list_videos_empty(tmp_path):
    """Return empty list when assets/videos does not exist."""
    with patch("resources.resources._ASSETS", str(tmp_path)):
        out = list_videos()
    data = json.loads(out)
    assert data["videos"] == []
    assert "path" in data
    assert data["count"] == 0


def test_list_videos_finds_mp4(tmp_path):
    """List .mp4 files in assets/videos."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir(parents=True)
    (videos_dir / "a.mp4").touch()
    (videos_dir / "b.mov").touch()
    (videos_dir / "skip.txt").touch()

    with patch("resources.resources._ASSETS", str(tmp_path)):
        out = list_videos()

    data = json.loads(out)
    assert set(data["videos"]) == {"a.mp4", "b.mov"}
    assert data["count"] == 2


def test_list_frames_empty(tmp_path):
    """Return empty folders when assets/frames does not exist."""
    with patch("resources.resources._ASSETS", str(tmp_path)):
        out = list_frames()
    data = json.loads(out)
    assert data["folders"] == []
    assert data["count"] == 0


def test_list_frames_finds_folders(tmp_path):
    """List subfolders in assets/frames."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(parents=True)
    (frames_dir / "out").mkdir()
    (frames_dir / "other").mkdir()

    with patch("resources.resources._ASSETS", str(tmp_path)):
        out = list_frames()

    data = json.loads(out)
    assert set(data["folders"]) == {"out", "other"}
    assert data["count"] == 2


def test_list_masks_empty(tmp_path):
    """Return empty inputs when assets/masks does not exist."""
    with patch("resources.resources._ASSETS", str(tmp_path)):
        out = list_masks()
    data = json.loads(out)
    assert data["inputs"] == []
    assert data["count"] == 0


def test_list_masks_finds_inputs(tmp_path):
    """List subfolders in assets/masks."""
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir(parents=True)
    (masks_dir / "out_point_0.50_0.50").mkdir()
    (masks_dir / "out_box_0.20_0.20_0.80_0.80").mkdir()

    with patch("resources.resources._ASSETS", str(tmp_path)):
        out = list_masks()

    data = json.loads(out)
    assert "out_point_0.50_0.50" in data["inputs"]
    assert "out_box_0.20_0.20_0.80_0.80" in data["inputs"]
    assert data["count"] == 2
