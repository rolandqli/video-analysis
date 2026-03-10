"""Unit tests for orchestrator prompts."""
import pytest

from orchestrator.prompts import extract_and_segment


def test_extract_and_segment_default_point():
    """Default point (0.5, 0.5) when neither point nor box given."""
    out = extract_and_segment("out.mp4")
    assert "point: [0.5, 0.5]" in out
    assert "box:" not in out or "point:" in out
    assert "out.mp4" in out
    assert "assets/videos/out.mp4" in out
    assert "assets/frames/out" in out


def test_extract_and_segment_custom_point():
    """Custom point is included."""
    out = extract_and_segment("video.mov", point=(0.3, 0.7))
    assert "point: [0.3, 0.7]" in out


def test_extract_and_segment_with_box():
    """Box is used instead of point when provided."""
    out = extract_and_segment("a.mp4", box=(0.2, 0.2, 0.8, 0.8))
    assert "box: [0.2, 0.2, 0.8, 0.8]" in out
    assert "point:" not in out or "point: [0.5, 0.5]" not in out


def test_extract_and_segment_stem_no_extension():
    """Stem derived from filename for frames path."""
    out = extract_and_segment("my_video.mp4")
    assert "assets/frames/my_video" in out
