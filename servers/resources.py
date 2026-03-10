"""MCP server for asset listing resources and tools."""

from fastmcp import FastMCP

from resources.resources import list_frames, list_masks, list_videos

mcp = FastMCP("resources")

mcp.tool()(list_videos)
mcp.tool()(list_frames)
mcp.tool()(list_masks)

mcp.resource("asset://videos", mime_type="application/json")(list_videos)
mcp.resource("asset://frames", mime_type="application/json")(list_frames)
mcp.resource("asset://masks", mime_type="application/json")(list_masks)

if __name__ == "__main__":
    mcp.run()
