# MCP Video Analysis

MCP (Model Context Protocol) servers for video frame extraction and image analysis.

## Project structure

```
apple/
├── server.py           # Video/image analysis MCP server
├── sam_server.py       # SAM 2 segmentation MCP server
├── tools/
│   ├── video.py        # video_to_frames
│   ├── image.py        # detect_objects, summarize_scene
│   └── sam.py          # segment_with_sam2
├── utils/
│   └── image_utils.py  # load_image, MIME_TYPES
└── ...
```

## Servers

### server.py — Video & Image Analysis

**Video:**
- **`video_to_frames(video_path, output_dir?, frame_interval?, image_format?)`** — Extract frames from a video to a folder of images (OpenCV)

**Image:**
- **`detect_objects(image_path)`** — Detect and describe objects in an image using OpenAI vision
- **`summarize_scene(scene_description)`** — Convert scene descriptions to structured metadata/tags

### sam_server.py — SAM 2 Segmentation

Uses [Meta's SAM 2](https://github.com/facebookresearch/sam2) with point prompts. No Hugging Face access request needed.

- **`segment_with_sam2(resource_path, point?, output_dir?, frame_index?, propagate?)`** — Segment objects in a video or folder of images. Uses point prompt (default: frame center). Saves masks and metadata to disk.

**Prerequisites:** Python 3.10+, PyTorch 2.5+, CUDA. Checkpoints auto-download from Hugging Face.

```bash
pip install 'git+https://github.com/facebookresearch/sam2.git'
```

### simple.py — Demo

- **`add(a, b)`** — Add two numbers (example MCP tool)

## Setup

Each MCP server uses its own virtual environment with isolated dependencies.

```bash
# Create per-service venvs (Python 3.10+ required)
PYTHON=/path/to/python3.10 ./scripts/setup_venvs.sh   # or python3.11, python3.12

# For SAM 2 (GPU/CUDA required):
.venv-sam2-segmentation/bin/pip install 'git+https://github.com/facebookresearch/sam2.git'

# Set API key for video-analysis (OpenAI)
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=your-key-here
```

**Requirements by service:**
- `requirements/video-analysis.txt` → `.venv-video-analysis`
- `requirements/sam2-segmentation.txt` → `.venv-sam2-segmentation`

## Configuration

Put the config in **project-level** `.cursor/mcp.json` (relative paths, no hardcoding):

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
    }
  }
}
```

Use your project’s absolute paths. The SAM 2 server requires a CUDA GPU.

## Run Manually

```bash
python server.py      # video-analysis server
python sam_server.py  # SAM 2 segmentation server (GPU required)
python simple.py      # demo server
```
