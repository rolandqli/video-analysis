"""MCP server for video and image analysis."""
import os

from dotenv import load_dotenv
from fastmcp import FastMCP
from openai import OpenAI

from tools.image import create_detect_objects, create_summarize_scene
from tools.video import video_to_frames

load_dotenv()

mcp = FastMCP("video-analysis")

mcp.tool()(video_to_frames)

if __name__ == "__main__":
    mcp.run()
