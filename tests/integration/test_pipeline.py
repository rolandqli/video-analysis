"""Integration tests for video → frames → segment pipeline."""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

cv2 = pytest.importorskip("cv2")
import numpy as np
import pytest

# Ensure project root on path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resources.resources import list_frames, list_masks, list_videos
from tools.sam import segment_with_sam2
from tools.video import video_to_frames


def _make_video(path: str, num_frames: int = 3) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, 30.0, (64, 48))
    for _ in range(num_frames):
        out.write(np.uint8(np.zeros((48, 64, 3)) + 128))
    out.release()


def _make_jpeg(path: Path, size: tuple[int, int] = (50, 50)) -> None:
    from PIL import Image
    img = Image.fromarray(np.zeros((*size, 3), dtype=np.uint8))
    img.save(str(path), "JPEG")


def _fake_sam2(mock_predictor):
    """Mock sam2 and torch only; use real matplotlib so savefig writes masked_images."""
    torch = MagicMock()
    torch.cuda.is_available.return_value = False
    torch.backends = MagicMock()
    torch.backends.mps = MagicMock()
    torch.backends.mps.is_available.return_value = False
    predictor_cls = MagicMock()
    predictor_cls.from_pretrained.return_value = mock_predictor
    return {
        "torch": torch,
        "sam2": MagicMock(),
        "sam2.sam2_video_predictor": MagicMock(SAM2VideoPredictor=predictor_cls),
    }


def test_video_to_frames_then_resources_list(tmp_path, monkeypatch):
    """After video_to_frames, list_videos and list_frames show the outputs."""
    monkeypatch.chdir(tmp_path)
    assets = tmp_path / "assets"
    monkeypatch.setattr("tools.video._ASSETS_DIR", str(assets))
    monkeypatch.setattr("resources.resources._ASSETS", str(assets))
    vid = tmp_path / "test.mp4"
    _make_video(str(vid))

    video_to_frames(str(vid))

    videos = json.loads(list_videos())
    assert "test.mp4" in videos["videos"]

    frames_data = json.loads(list_frames())
    assert "test" in frames_data["folders"]


def test_full_pipeline_video_frames_segment(tmp_path, monkeypatch):
    """Full pipeline: extract frames, then segment (with mocked SAM 2)."""
    monkeypatch.chdir(tmp_path)
    assets = tmp_path / "assets"
    monkeypatch.setattr("tools.video._ASSETS_DIR", str(assets))
    monkeypatch.setattr("resources.resources._ASSETS", str(assets))
    vid = tmp_path / "sample.mp4"
    _make_video(str(vid), num_frames=2)

    # Step 1: extract frames
    extract_result = video_to_frames(str(vid))
    frames_dir = Path(extract_result["output_dir"])
    assert frames_dir.exists()
    assert len(list(frames_dir.glob("frame_*.jpg"))) == 2

    # Step 2: segment (mock SAM 2)
    mock_predictor = MagicMock()
    mock_predictor.init_state.return_value = MagicMock()
    fake_mask = MagicMock()
    _arr = np.zeros((48, 64), dtype=bool)
    _arr[10:40, 10:50] = True  # non-empty mask
    gt_result = MagicMock()
    gt_result.cpu.return_value.numpy.return_value = _arr
    fake_mask.__gt__ = lambda s, o: gt_result
    mock_predictor.add_new_points_or_box.return_value = (0, [1], [fake_mask])
    # propagate_in_video yields (frame_idx, obj_ids, mask_logits) for each extra frame
    mock_predictor.propagate_in_video.return_value = iter([
        (1, [1], [fake_mask]),
    ])

    # Reset cached predictor so we get our mock (previous tests may have set it)
    import tools.sam as sam_mod
    sam_mod._PREDICTOR = None

    with patch.dict(sys.modules, _fake_sam2(mock_predictor)):
        seg_result = segment_with_sam2(
            str(frames_dir),
            point=(0.5, 0.5),
            propagate=True,
        )

    assert seg_result["num_frames"] == 2
    assert "saved_frames" in seg_result
    assert "saved_masks" in seg_result
    assert len(seg_result["saved_masks"]) == 2
    assert len(seg_result["saved_frames"]) == 2
    out = Path(seg_result["output_dir"])
    assert (out / "masks").exists()
    assert (out / "masked_images").exists()
    assert len(list((out / "masks").glob("frame_*.png"))) == 2
    assert len(list((out / "masked_images").glob("frame_*.png"))) == 2
    assert all("masked_images" in p for p in seg_result["saved_frames"])

    # Step 3: list_masks shows the new output (resources use cwd/assets)
    monkeypatch.setattr("resources.resources._ASSETS", str(assets))
    masks_data = json.loads(list_masks())
    assert len(masks_data["inputs"]) >= 1
    assert any("sample" in inp for inp in masks_data["inputs"])
