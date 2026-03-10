# MCP Video Analysis

MCP (Model Context Protocol) servers for image analysis and a simple demo.

## Servers

### server.py — Video Analysis

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
