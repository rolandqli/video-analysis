"""Unit tests for video_to_frames."""
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
import pytest

from tools.video import video_to_frames


def _make_video(path: str, num_frames: int = 5, fps: float = 30.0) -> None:
    """Create a minimal test video file."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (64, 48))
    for _ in range(num_frames):
        frame = np.uint8(np.zeros((48, 64, 3)) + 128)
        out.write(frame)
    out.release()


def test_video_file_not_found():
    """Raise FileNotFoundError when video does not exist."""
    with pytest.raises(FileNotFoundError, match="Video file not found"):
        video_to_frames("/nonexistent/video.mp4")


def test_invalid_image_format(tmp_path):
    """Raise ValueError for unsupported image format."""
    vid = tmp_path / "test.mp4"
    _make_video(str(vid))
    with pytest.raises(ValueError, match="image_format must be 'jpg' or 'png'"):
        video_to_frames(str(vid), image_format="gif")


def test_extract_frames_default_output(tmp_path, monkeypatch):
    """Extract frames and use assets/frames/<stem> by default."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("tools.video._ASSETS_DIR", str(tmp_path / "assets"))
    vid = tmp_path / "sample.mp4"
    _make_video(str(vid), num_frames=3)

    result = video_to_frames(str(vid))

    assert result["num_frames"] == 3
    assert result["total_video_frames"] == 3
    assert result["fps"] == 30.0
    assert "output_dir" in result
    assert "video_copy_path" in result
    frames_dir = Path(result["output_dir"])
    assert frames_dir.name == "sample"
    assert (frames_dir.parent / frames_dir.name) == frames_dir
    assert len(list(frames_dir.glob("frame_*.jpg"))) == 3


def test_extract_frames_explicit_output_dir(tmp_path, monkeypatch):
    """Explicit output_dir is used."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("tools.video._ASSETS_DIR", str(tmp_path / "assets"))
    vid = tmp_path / "a.mp4"
    _make_video(str(vid), num_frames=2)
    custom = tmp_path / "my_frames"

    result = video_to_frames(str(vid), output_dir=str(custom))

    assert result["output_dir"] == str(custom.resolve())
    assert len(list(custom.glob("frame_*.jpg"))) == 2


def test_extract_every_nth_frame(tmp_path, monkeypatch):
    """frame_interval extracts every N-th frame."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("tools.video._ASSETS_DIR", str(tmp_path / "assets"))
    vid = tmp_path / "b.mp4"
    _make_video(str(vid), num_frames=6)
    out = tmp_path / "out"
    out.mkdir()

    result = video_to_frames(str(vid), output_dir=str(out), frame_interval=2)

    assert result["num_frames"] == 3  # frames 0, 2, 4
    assert len(list(out.glob("frame_*.jpg"))) == 3


def test_video_copied_to_assets_videos(tmp_path, monkeypatch):
    """Video is copied to assets/videos/."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("tools.video._ASSETS_DIR", str(tmp_path / "assets"))
    vid = tmp_path / "foo.mp4"
    _make_video(str(vid))

    result = video_to_frames(str(vid))

    copy_path = Path(result["video_copy_path"])
    assert copy_path.name == "foo.mp4"
    assert "videos" in str(copy_path)
    assert copy_path.exists()
