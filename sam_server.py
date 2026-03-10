"""MCP server for SAM 2 (Segment Anything Model 2) segmentation."""

from fastmcp import FastMCP

from resources.assets import list_masks
from tools.sam import segment_with_sam2

mcp = FastMCP("sam2-segmentation")

mcp.tool()(segment_with_sam2)

# Resource
mcp.resource("asset://masks", mime_type="application/json")(list_masks)

if __name__ == "__main__":
    mcp.run()
