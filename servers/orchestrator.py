"""MCP server for orchestrator prompts (multi-step workflow guidance)."""

from fastmcp import FastMCP

from orchestrator.prompts import extract_and_segment

mcp = FastMCP("orchestrator")

mcp.prompt()(extract_and_segment)

if __name__ == "__main__":
    mcp.run()
