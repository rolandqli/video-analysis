"""MCP server for SAM 2 (Segment Anything Model 2) segmentation."""

from fastmcp import FastMCP

from tools.sam import segment_with_sam2

mcp = FastMCP("sam2-segmentation")

mcp.tool()(segment_with_sam2)

if __name__ == "__main__":
    mcp.run()
