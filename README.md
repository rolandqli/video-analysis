# MCP Video Analysis

MCP (Model Context Protocol) servers for video frame extraction and image analysis.

## Project structure

```
apple/
├── server.py           # MCP entry point, registers tools
├── tools/
│   ├── video.py        # video_to_frames
│   └── image.py        # detect_objects, summarize_scene
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

### simple.py — Demo

- **`add(a, b)`** — Add two numbers (example MCP tool)

## Setup

```bash
# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set API key
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=your-key-here
```

## Configuration

Add to `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "video-analysis": {
      "command": "python",
      "args": ["/path/to/this/project/server.py"],
      "cwd": "/path/to/this/project"
    }
  }
}
```

Use your project’s absolute paths. With a virtualenv, set `command` to `.venv/bin/python`.

## Run Manually

```bash
python server.py   # video-analysis server
python simple.py   # demo server
```
