"""MCP server for orchestrator prompts (multi-step workflow guidance)."""

from fastmcp import FastMCP

from orchestrator.prompts import extract_and_segment

mcp = FastMCP("orchestrator")

# Register prompts (MCP protocol prompts/get)
mcp.prompt()(extract_and_segment)

# Registry for get_prompt tool
_PROMPTS = {"extract_and_segment": extract_and_segment}


@mcp.tool()
def get_prompt(
    name: str,
    video: str | None = None,
    point: list[float] | None = None,
    box: list[float] | None = None,
) -> str:
    """
    Retrieve a rendered prompt by name. Returns the prompt text with arguments applied.
    Use this to fetch workflow instructions before executing tools.

    Args:
        name: Prompt name (e.g. "extract_and_segment").
        video: For extract_and_segment: video filename (e.g. "out.mp4").
        point: For extract_and_segment: [x, y] normalized 0-1. Default [0.5, 0.5].
        box: For extract_and_segment: optional [x1, y1, x2, y2] normalized 0-1.
            When set, used instead of point for SAM 2.

    Returns:
        Rendered prompt text (instructions to follow).
    """
    if name not in _PROMPTS:
        available = ", ".join(sorted(_PROMPTS))
        return f"Unknown prompt: {name}. Available: {available}"
    fn = _PROMPTS[name]
    if name == "extract_and_segment":
        if video is None:
            return "extract_and_segment requires 'video' argument."
        pt = tuple(point) if point and len(point) == 2 else None
        bx = tuple(box) if box and len(box) == 4 else None
        return fn(video=video, point=pt, box=bx)
    return fn()
