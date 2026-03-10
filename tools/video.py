"""Video processing tools."""
import os

import cv2


def video_to_frames(
    video_path: str,
    output_dir: str | None = None,
    frame_interval: int = 1,
    image_format: str = "jpg",
) -> dict:
    """
    Extract frames from a video file and save them as images in a folder.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory to save frames. If None, creates <video_stem>_frames
            in the same directory as the video.
        frame_interval: Extract every N-th frame (1 = all frames).
        image_format: Output format for frames (jpg or png).

    Returns:
        Dict with output_dir, num_frames, total_video_frames, and fps.
    """
    if image_format.lower() not in ("jpg", "jpeg", "png"):
        raise ValueError("image_format must be 'jpg' or 'png'")

    video_path = os.path.abspath(video_path)
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if output_dir is None:
        stem = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(os.path.dirname(video_path), f"{stem}_frames")

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    ext = ".jpg" if image_format.lower() in ("jpg", "jpeg") else ".png"
    num_saved = 0
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                out_path = os.path.join(output_dir, f"frame_{num_saved:06d}{ext}")
                cv2.imwrite(out_path, frame)
                num_saved += 1

            frame_idx += 1
    finally:
        cap.release()

    return {
        "output_dir": output_dir,
        "num_frames": num_saved,
        "total_video_frames": total_frames,
        "fps": round(fps, 2),
    }
