"""MCP server for video and image analysis."""
from dotenv import load_dotenv
from fastmcp import FastMCP
from openai import OpenAI
import os

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

if __name__ == "__main__":
    mcp.run()
