from __future__ import annotations

"""
Grounding DINO integration as an MCP tool.

This wraps the official Grounding DINO repo
https://github.com/IDEA-Research/GroundingDINO
for open-set object detection with text prompts.
"""

import json
import os
from typing import Any, List

import numpy as np


def detect_with_groundingdino(
    image_path: str,
    text_prompt: str,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
) -> dict[str, Any]:
    """
    Run Grounding DINO open-set object detection on an image.

    Args:
        image_path: Path to the input image.
        text_prompt: Text prompt describing objects to detect, e.g. "person . dog . chair .".
        box_threshold: Minimum confidence threshold for boxes.
        text_threshold: Minimum text similarity threshold.

    Returns:
        Dict containing boxes, logits, phrases and the image size.
    """
    try:
        from groundingdino.util.inference import load_model, load_image, predict  # type: ignore
    except ImportError as e:
        raise ImportError(
            "GroundingDINO is not installed. Install with:\n"
            "  pip install 'git+https://github.com/IDEA-Research/GroundingDINO.git'\n"
            "and ensure it is available in the groundingdino server virtualenv."
        ) from e

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Use the default SwinT config + checkpoint from the README.
    # Expect environment variables to point to the config and weights if customized.
    cfg = os.getenv(
        "GROUNDINGDINO_CONFIG",
        "groundingdino/config/GroundingDINO_SwinT_OGC.py",
    )
    weights = os.getenv(
        "GROUNDINGDINO_WEIGHTS",
        "weights/groundingdino_swint_ogc.pth",
    )

    model = load_model(cfg, weights)
    image_source, image = load_image(image_path)

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    # Convert numpy arrays to lists for JSON-serializable output
    boxes_list: List[List[float]] = boxes.tolist() if isinstance(boxes, np.ndarray) else boxes  # type: ignore[arg-type]
    logits_list: List[float] = logits.tolist() if isinstance(logits, np.ndarray) else logits  # type: ignore[arg-type]

    result = {
        "image_path": os.path.abspath(image_path),
        "image_size": {
            "height": int(image_source.shape[0]),
            "width": int(image_source.shape[1]),
        },
        "text_prompt": text_prompt,
        "box_threshold": box_threshold,
        "text_threshold": text_threshold,
        "boxes": boxes_list,
        "logits": logits_list,
        "phrases": phrases,
    }

    # Store metadata JSON alongside the source image
    image_dir = os.path.dirname(os.path.abspath(image_path))
    stem, _ = os.path.splitext(os.path.basename(image_path))
    metadata_path = os.path.join(image_dir, f"{stem}.groundingdino.json")
    with open(metadata_path, "w") as f:
        json.dump(result, f, indent=2)
    result["metadata_path"] = metadata_path

    return result

