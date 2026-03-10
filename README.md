# MCP Video Analysis

MCP (Model Context Protocol) servers for video frame extraction, image analysis, SAM 2 segmentation, and orchestrated workflows.

## Project structure

```
video-analysis/
├── servers/                  # MCP server entrypoints
│   ├── video_analysis.py     # Video/image analysis
│   ├── sam2_segmentation.py  # SAM 2 segmentation
│   ├── resources.py          # Asset listing resources & tools
│   ├── orchestrator.py       # Workflow prompts
│   └── grounding.py          # Grounding DINO (WIP)
├── tools/                    # Tool implementations
│   ├── video.py              # video_to_frames
│   ├── image.py              # detect_objects, summarize_scene
│   ├── sam.py                # segment_with_sam2
│   └── groundingdino.py      # detect_with_groundingdino
├── resources/                # MCP resource functions
│   └── resources.py          # list_videos, list_frames, list_masks
├── orchestrator/             # Workflow prompts
│   └── prompts.py            # extract_and_segment
├── assets/                   # Output assets (auto-created)
│   ├── videos/               # Video copies (from video_to_frames)
│   ├── frames/               # Extracted frames (one folder per video)
│   └── masks/                # SAM 2 masks (one folder per input)
├── requirements/             # Per-service dependencies
├── scripts/
│   ├── mcp-launch.sh         # Launch MCP servers
│   └── setup_venvs.sh        # Create venvs
└── .cursor/mcp.json          # MCP server config
```

## Servers

### video-analysis — Video & Image

**Tools:**
- **`video_to_frames(video_path, output_dir?, frame_interval?, image_format?)`** — Extract frames from a video. Copies video to `assets/videos/`, saves frames to `assets/frames/<stem>/`.

**Image (OpenAI):**
- **`detect_objects(image_path)`** — Detect and describe objects in an image
- **`summarize_scene(scene_description)`** — Convert scene descriptions to structured metadata

### sam2-segmentation — SAM 2

Uses [Meta's SAM 2](https://github.com/facebookresearch/sam2) with point prompts. Model is loaded once at startup.

- **`segment_with_sam2(resource_path, point?, box?, output_dir?, frame_index?, propagate?)`** — Segment objects in a video or folder of images. Point or box normalized 0–1. Saves to `assets/masks/<input_name>_<prompt>/`: `masks/` (raw binary masks) and `masked_images/` (overlay visualizations).

**Prerequisites:** Python 3.10+, PyTorch. On Mac without CUDA: `SAM2_BUILD_CUDA=0 pip install 'git+https://github.com/facebookresearch/sam2.git'`

### resources — Asset Listings

**Tools & Resources:**
- `list_videos` / `asset://videos` — List videos in `assets/videos/`
- `list_frames` / `asset://frames` — List frame folders in `assets/frames/`
- `list_masks` / `asset://masks` — List mask outputs in `assets/masks/`

### orchestrator — Workflow Prompts

**Prompts:**
- **`extract_and_segment(video, point_x?, point_y?)`** — Returns instructions to: discover video via `asset://videos`, run `video_to_frames`, then `segment_with_sam2` with the given point. Use this prompt to drive the full pipeline.

### groundingdino — Grounding DINO *(work in progress)*

[Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) for open-set object detection with text prompts. Server exists (`servers/grounding.py`) but is not wired into `mcp-launch.sh` or `.cursor/mcp.json` yet.

- **`groundingdino_detect(image_path, text_prompt, box_threshold?, text_threshold?)`** — Detect objects in an image by text prompt (e.g. `"person . dog . chair ."`).

## Setup

```bash
# Create per-service venvs (Python 3.10+ required)
./scripts/setup_venvs.sh

# SAM 2 (Mac without CUDA):
SAM2_BUILD_CUDA=0 .venv-sam2-segmentation/bin/pip install 'git+https://github.com/facebookresearch/sam2.git'

# video-analysis (OpenAI):
cp .env.example .env
# Add OPENAI_API_KEY=your-key
```

**Venvs:**
- `.venv-video-analysis` — video/image tools
- `.venv-sam2-segmentation` — SAM 2
- `.venv-resources` — asset listings
- `.venv-orchestrator` — prompts

## Configuration

`.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "video-analysis": {
      "command": "scripts/mcp-launch.sh",
      "args": ["video-analysis"],
      "cwd": ".."
    },
    "sam2-segmentation": {
      "command": "scripts/mcp-launch.sh",
      "args": ["sam2-segmentation"],
      "cwd": ".."
    },
    "resources": {
      "command": "scripts/mcp-launch.sh",
      "args": ["resources"],
      "cwd": ".."
    },
    "orchestrator": {
      "command": "scripts/mcp-launch.sh",
      "args": ["orchestrator"],
      "cwd": ".."
    }
  }
}
```

## Run Manually

```bash
.venv-video-analysis/bin/python servers/video_analysis.py
.venv-sam2-segmentation/bin/python servers/sam2_segmentation.py
.venv-resources/bin/python servers/resources.py
.venv-orchestrator/bin/python servers/orchestrator.py
```

Or use `scripts/mcp-launch.sh <server-name>`.

## Tests

```bash
# Create test venv (optional; or use project venv with deps)
./scripts/setup_venvs.sh   # includes .venv-test

# Run tests
.venv-test/bin/pytest   # or: python3 -m pytest tests/
```

**Unit tests:** `test_video`, `test_resources`, `test_prompts`, `test_sam`  
**Integration tests:** `tests/integration/test_pipeline` (video → frames → segment)
