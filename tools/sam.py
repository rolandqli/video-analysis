"""SAM 2 (Segment Anything Model 2) integration for video/image segmentation."""

from __future__ import annotations

import glob
import json
import os
import shutil
import sys
import tempfile
from typing import Any

import numpy as np


def _prepare_frame_folder_for_sam2(folder_path: str) -> tuple[str, Any]:
    """
    Ensure folder has SAM 2-compatible frame names (stem must parse as int, e.g. 000000.jpg).
    Returns (path_to_use, cleanup_func). Call cleanup_func when done to remove temp dir.
    """
    paths = sorted(
        glob.glob(os.path.join(folder_path, "*.jpg")) + glob.glob(os.path.join(folder_path, "*.jpeg"))
    )
    if not paths:
        return folder_path, lambda: None
    first_stem = os.path.splitext(os.path.basename(paths[0]))[0]
    try:
        int(first_stem)
        return folder_path, lambda: None  # Already compatible
    except ValueError:
        pass
    tmpdir = tempfile.mkdtemp(prefix="sam2_frames_")
    for i, src in enumerate(paths):
        ext = os.path.splitext(src)[1]
        dst = os.path.join(tmpdir, f"{i:06d}{ext}")
        os.symlink(os.path.abspath(src), dst)
    return tmpdir, lambda: shutil.rmtree(tmpdir, ignore_errors=True)


def _get_device() -> str:
    """Pick device: CUDA if available, else MPS on Mac, else CPU."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if sys.platform == "darwin" and getattr(torch.backends, "mps", None) is not None:
        if torch.backends.mps.is_available():
            return "mps"
    return "cpu"


_PREDICTOR = None


def _get_predictor():
    """Load SAM 2 predictor once and reuse."""
    global _PREDICTOR
    if _PREDICTOR is None:
        try:
            from sam2.sam2_video_predictor import SAM2VideoPredictor
        except ImportError as e:
            raise ImportError(
                "SAM 2 is not installed. Install with:\n"
                "  pip install 'git+https://github.com/facebookresearch/sam2.git'\n"
                "Checkpoints download automatically from Hugging Face (no access request needed)."
            ) from e
        _PREDICTOR = SAM2VideoPredictor.from_pretrained(
            "facebook/sam2-hiera-tiny", device=_get_device()
        )
    return _PREDICTOR


def segment_with_sam2(
    resource_path: str,
    point: tuple[float, float] | None = None,
    box: tuple[float, float, float, float] | None = None,
    output_dir: str | None = None,
    frame_index: int = 0,
    propagate: bool = True,
) -> dict:
    """
    Run SAM 2 segmentation on a video file or folder of images.

    Uses Meta's Segment Anything Model 2 with point and/or box prompts (no Hugging Face access).
    Supports both MP4 video files and folders of JPEG frames.
    Provide point and/or box; both are normalized 0-1. If neither given, uses center (0.5, 0.5).
    Box format: (x1, y1, x2, y2) top-left to bottom-right.

    Prerequisites: Python 3.10+, PyTorch 2.5+, CUDA. Install: pip install 'git+https://github.com/facebookresearch/sam2.git'

    Args:
        resource_path: Path to an MP4 video file or a folder of JPEG images.
        point: (x, y) point prompt normalized 0-1. Default (0.5, 0.5) if no box.
        box: (x1, y1, x2, y2) bounding box normalized 0-1. Takes precedence over point when set.
        output_dir: Directory to save outputs. Defaults to assets/masks/<input_name>.
        frame_index: Frame to add the prompt on.
        propagate: If True, propagate segmentation to all frames.

    Returns:
        Dict with output_dir, num_frames, saved file paths.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    resource_path = os.path.abspath(resource_path)
    if not os.path.exists(resource_path):
        raise FileNotFoundError(f"Resource not found: {resource_path}")

    _assets_dir = os.path.join(os.getcwd(), "assets")
    _masks_base = os.path.join(_assets_dir, "masks")

    # Compute prompt for output folder name
    if box is not None:
        x1, y1, x2, y2 = (
            max(0.0, min(1.0, box[0])),
            max(0.0, min(1.0, box[1])),
            max(0.0, min(1.0, box[2])),
            max(0.0, min(1.0, box[3])),
        )
        prompt_desc = [x1, y1, x2, y2]
        prompt_suffix = f"box_{x1:.2f}_{y1:.2f}_{x2:.2f}_{y2:.2f}"
    else:
        px, py = point or (0.5, 0.5)
        px, py = max(0.0, min(1.0, px)), max(0.0, min(1.0, py))
        prompt_desc = [px, py]
        prompt_suffix = f"point_{px:.2f}_{py:.2f}"

    if output_dir is None:
        base = os.path.basename(resource_path.rstrip("/\\"))
        input_name = os.path.splitext(base)[0] if os.path.isfile(resource_path) else base
        output_dir = os.path.join(_masks_base, f"{input_name}_{prompt_suffix}")

    output_dir = os.path.abspath(output_dir)
    masks_dir = os.path.join(output_dir, "masks")
    masked_images_dir = os.path.join(output_dir, "masked_images")
    metadata_dir = os.path.join(output_dir, "metadata")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(masked_images_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    predictor = _get_predictor()

    # SAM 2 requires frame filenames with integer stems (e.g. 000000.jpg); create temp dir if needed
    if os.path.isdir(resource_path):
        sam2_path, cleanup = _prepare_frame_folder_for_sam2(resource_path)
    else:
        sam2_path = resource_path
        cleanup = lambda: None
    try:
        inference_state = predictor.init_state(sam2_path)
    finally:
        cleanup()

    # Get first frame size for pixel coords
    if os.path.isfile(resource_path):
        import cv2
        cap = cv2.VideoCapture(resource_path)
        _, frame = cap.read()
        h, w = frame.shape[:2]
        cap.release()
    else:
        paths = sorted(
            glob.glob(os.path.join(resource_path, "*.jpg"))
            + glob.glob(os.path.join(resource_path, "*.jpeg"))
        )
        from PIL import Image
        img = np.array(Image.open(paths[0]))
        h, w = img.shape[:2]

    if box is not None:
        box_px = np.array(
            [prompt_desc[0] * w, prompt_desc[1] * h, prompt_desc[2] * w, prompt_desc[3] * h],
            dtype=np.float32,
        )
        add_kwargs: dict[str, Any] = {"box": box_px}
    else:
        px, py = prompt_desc[0], prompt_desc[1]
        pt_x = int(px * w)
        pt_y = int(py * h)
        points = np.array([[pt_x, pt_y]], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)
        add_kwargs = {"points": points, "labels": labels}

    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=frame_index,
        obj_id=1,
        **add_kwargs,
    )

    outputs_per_frame: dict[int, Any] = {frame_index: (out_obj_ids, out_mask_logits)}

    if propagate:
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
            outputs_per_frame[frame_idx] = (obj_ids, mask_logits)

    # Load frames and save visualizations
    video_frames: dict[int, np.ndarray] = {}
    if os.path.isfile(resource_path):
        import cv2
        cap = cv2.VideoCapture(resource_path)
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            video_frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            i += 1
        cap.release()
    else:
        from PIL import Image
        paths = sorted(
            glob.glob(os.path.join(resource_path, "*.jpg"))
            + glob.glob(os.path.join(resource_path, "*.jpeg"))
        )
        for i, p in enumerate(paths):
            video_frames[i] = np.array(Image.open(p).convert("RGB"))

    import cv2 as _cv2

    saved_masks: list[str] = []
    saved_frames: list[str] = []
    for idx in sorted(outputs_per_frame.keys()):
        if idx not in video_frames:
            continue
        try:
            obj_ids, mask_logits = outputs_per_frame[idx]
            # Build combined binary mask (union of all object masks)
            combined_mask = None
            mask_list = []
            for oid, mlog in zip(obj_ids, mask_logits):
                mask = (mlog > 0.0).cpu().numpy() if hasattr(mlog, "cpu") else (mlog > 0.0)
                if mask.ndim == 3:
                    mask = mask[0]
                mask_list.append(mask)
                if combined_mask is None:
                    combined_mask = mask.copy()
                else:
                    combined_mask = np.logical_or(combined_mask, mask)
            if combined_mask is None:
                combined_mask = np.zeros(video_frames[idx].shape[:2], dtype=bool)
            # Save actual mask (0/255) to masks/
            mask_uint8 = np.where(combined_mask, 255, 0).astype(np.uint8)
            mask_path = os.path.join(masks_dir, f"frame_{idx:06d}.png")
            _cv2.imwrite(mask_path, mask_uint8)
            saved_masks.append(mask_path)
            # Save overlay visualization to masked_images/
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.imshow(video_frames[idx])
            for mask in mask_list:
                mask_rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
                mask_rgba[mask, :] = [255, 0, 0, 100]
                ax.imshow(mask_rgba)
            ax.set_title(f"Frame {idx}")
            masked_img_path = os.path.join(masked_images_dir, f"frame_{idx:06d}.png")
            plt.savefig(masked_img_path, bbox_inches="tight", dpi=150)
            plt.close("all")
            saved_frames.append(masked_img_path)
        except Exception:
            plt.close("all")

    meta = {
        "resource_path": resource_path,
        "point": prompt_desc if box is None else None,
        "box": prompt_desc if box is not None else None,
        "frame_index": frame_index,
        "propagate": propagate,
        "num_frames": len(outputs_per_frame),
    }
    meta_path = os.path.join(metadata_dir, "run.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "output_dir": output_dir,
        "num_frames": len(outputs_per_frame),
        "metadata_path": meta_path,
        "saved_masks": saved_masks,
        "saved_frames": saved_frames,
        "point": prompt_desc if box is None else None,
        "box": prompt_desc if box is not None else None,
    }
