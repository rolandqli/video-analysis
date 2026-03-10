"""MCP server for video and image analysis."""
import os

from dotenv import load_dotenv
from fastmcp import FastMCP
from openai import OpenAI

from resources.assets import list_frames, list_videos
from tools.image import create_detect_objects, create_summarize_scene
from tools.video import video_to_frames

load_dotenv()

mcp = FastMCP("video-analysis")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Register video tools
mcp.tool()(video_to_frames)

# Register image tools (require OpenAI client)
mcp.tool()(create_detect_objects(client))
mcp.tool()(create_summarize_scene(client))

# Resources
mcp.resource("asset://videos", mime_type="application/json")(list_videos)
mcp.resource("asset://frames", mime_type="application/json")(list_frames)

if __name__ == "__main__":
    mcp.run()
