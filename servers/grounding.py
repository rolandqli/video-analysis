"""MCP server for Grounding DINO open-set object detection."""

from fastmcp import FastMCP

from tools.groundingdino import detect_with_groundingdino

mcp = FastMCP("groundingdino")


@mcp.tool()
def groundingdino_detect(
    image_path: str,
    text_prompt: str,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
):
    """Run Grounding DINO on an image with a text prompt."""
    return detect_with_groundingdino(
        image_path=image_path,
        text_prompt=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )


if __name__ == "__main__":
    mcp.run()
