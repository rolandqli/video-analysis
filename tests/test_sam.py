"""Unit tests for segment_with_sam2."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tools.sam import segment_with_sam2


def _make_test_jpeg(path: Path, size: tuple[int, int] = (50, 50)) -> None:
    """Create a minimal valid JPEG file."""
    from PIL import Image
    img = Image.fromarray(np.zeros((*size, 3), dtype=np.uint8))
    img.save(str(path), "JPEG")


def _make_fake_sam2_modules(mock_predictor: MagicMock) -> dict:
    """Fake sam2, torch, matplotlib for imports."""
    torch = MagicMock()
    torch.cuda.is_available.return_value = False
    if hasattr(torch, "backends"):
        torch.backends.mps = MagicMock()
        torch.backends.mps.is_available.return_value = False
    mpl = MagicMock()
    mpl.use = MagicMock()
    mpl_plt = MagicMock()
    mpl_plt.savefig = MagicMock()
    mpl_plt.close = MagicMock()
    fig, ax = MagicMock(), MagicMock()
    ax.imshow = MagicMock()
    ax.set_title = MagicMock()
    mpl_plt.subplots.return_value = (fig, ax)
    predictor_cls = MagicMock()
    predictor_cls.from_pretrained.return_value = mock_predictor
    return {
        "torch": torch,
        "sam2": MagicMock(),
        "sam2.sam2_video_predictor": MagicMock(SAM2VideoPredictor=predictor_cls),
        "matplotlib": mpl,
        "matplotlib.pyplot": mpl_plt,
    }


def test_import_error_when_sam2_not_installed(tmp_path):
    """Raise ImportError when SAM 2 predictor cannot be loaded."""
    frames = tmp_path / "frames"
    frames.mkdir()
    _make_test_jpeg(frames / "frame_000000.jpg")

    mpl = MagicMock()
    mpl.use = MagicMock()
    plt = MagicMock()
    plt.subplots = MagicMock(return_value=(MagicMock(), MagicMock()))
    plt.savefig = MagicMock()
    plt.close = MagicMock()
    fake_mpl = {"matplotlib": mpl, "matplotlib.pyplot": plt}

    with patch.dict(sys.modules, fake_mpl):
        with patch("tools.sam._get_predictor") as mock_get:
            mock_get.side_effect = ImportError(
                "SAM 2 is not installed. Install with: "
                "pip install 'git+https://github.com/facebookresearch/sam2.git'"
            )
            with pytest.raises(ImportError) as exc:
                segment_with_sam2(str(frames))
    assert "SAM 2" in str(exc.value)


def test_file_not_found(tmp_path):
    """Raise FileNotFoundError when resource_path does not exist."""
    mock_predictor = MagicMock()
    mods = _make_fake_sam2_modules(mock_predictor)

    with patch.dict(sys.modules, mods):
        with pytest.raises(FileNotFoundError, match="Resource not found"):
            segment_with_sam2(str(tmp_path / "missing"))


def test_output_dir_includes_point_suffix(tmp_path, monkeypatch):
    """Output dir defaults to assets/masks/<input>_point_0.50_0.50 when using point."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "assets" / "masks").mkdir(parents=True)
    frames = tmp_path / "my_frames"
    frames.mkdir()
    _make_test_jpeg(frames / "frame_000000.jpg")

    mock_predictor = MagicMock()
    mock_predictor.init_state.return_value = MagicMock()
    fake_mask = MagicMock()
    fake_mask.cpu.return_value.numpy.return_value = np.zeros((50, 50), dtype=bool)
    mock_predictor.add_new_points_or_box.return_value = (0, [1], [fake_mask])
    mock_predictor.propagate_in_video.return_value = iter([])

    with patch.dict(sys.modules, _make_fake_sam2_modules(mock_predictor)):
        result = segment_with_sam2(str(frames), point=(0.5, 0.5), propagate=False)

    assert "my_frames_point_0.50_0.50" in result["output_dir"]
    assert result["point"] == [0.5, 0.5]
    assert result["box"] is None
    assert result["num_frames"] == 1


def test_output_dir_includes_box_suffix(tmp_path, monkeypatch):
    """Output dir defaults to assets/masks/<input>_box_* when using box."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "assets" / "masks").mkdir(parents=True)
    frames = tmp_path / "foo"
    frames.mkdir()
    _make_test_jpeg(frames / "frame_000000.jpg")

    mock_predictor = MagicMock()
    mock_predictor.init_state.return_value = MagicMock()
    fake_mask = MagicMock()
    fake_mask.cpu.return_value.numpy.return_value = np.zeros((50, 50), dtype=bool)
    mock_predictor.add_new_points_or_box.return_value = (0, [1], [fake_mask])
    mock_predictor.propagate_in_video.return_value = iter([])

    with patch.dict(sys.modules, _make_fake_sam2_modules(mock_predictor)):
        result = segment_with_sam2(str(frames), box=(0.2, 0.2, 0.8, 0.8), propagate=False)

    assert "foo_box_0.20_0.20_0.80_0.80" in result["output_dir"]
    assert result["box"] == [0.2, 0.2, 0.8, 0.8]
    assert result["point"] is None


def test_explicit_output_dir(tmp_path, monkeypatch):
    """Explicit output_dir is used."""
    monkeypatch.chdir(tmp_path)
    frames = tmp_path / "frames"
    frames.mkdir()
    _make_test_jpeg(frames / "frame_000000.jpg")
    custom = tmp_path / "custom_out"

    mock_predictor = MagicMock()
    mock_predictor.init_state.return_value = MagicMock()
    fake_mask = MagicMock()
    fake_mask.cpu.return_value.numpy.return_value = np.zeros((50, 50), dtype=bool)
    mock_predictor.add_new_points_or_box.return_value = (0, [1], [fake_mask])
    mock_predictor.propagate_in_video.return_value = iter([])

    with patch.dict(sys.modules, _make_fake_sam2_modules(mock_predictor)):
        result = segment_with_sam2(str(frames), output_dir=str(custom), propagate=False)

    assert result["output_dir"] == str(custom.resolve())
    assert (custom / "metadata" / "run.json").exists()
