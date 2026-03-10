"""Orchestrator prompts for multi-tool workflows."""


def extract_and_segment(video: str, point_x: float = 0.5, point_y: float = 0.5) -> str:
    """Instructions for extract-frames-then-SAM2-segment workflow."""
    stem = video.rsplit(".", 1)[0] if "." in video else video
    return f"""Run this workflow in order:

1. **Discover the video**: Use the asset://videos resource to see available videos. If "{video}" is listed, the path is assets/videos/{video}.

2. **Extract frames**: Call video_to_frames with video_path="assets/videos/{video}" (or the correct path if different). Frames will be saved to assets/frames/<video_stem>/.

3. **Segment with SAM 2**: Call segment_with_sam2 with:
   - resource_path: the frames folder from step 2 (e.g. assets/frames/{stem})
   - point: [{point_x}, {point_y}]

4. Report the output paths (masks go to assets/masks/<input_name>/).
"""
