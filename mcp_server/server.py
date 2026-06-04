"""
Agentic Perception Platform — MCP server (Phase 0).

This is the *tool API surface* the agent talks to. In Phase 1 we register the
real perception tools here (detect_objects / segment_scene / bev_map / ...),
each wrapping a trained model from demo/pipeline.py and returning LLM-readable
JSON instead of raw tensors.

Phase 0 ships exactly one trivial tool — `ping` — so we can prove the full
MCP round-trip (client launches this server over stdio, lists its tools, calls
one, gets a result back) before any model code is involved.

Run standalone (for a quick sanity check):
    python -m mcp_server.server
The client in agent/mcp_client.py normally launches it for you.
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

# FastMCP introspects each @tool function's type hints + docstring to build the
# JSON Schema the LLM sees. Name it so the agent's tool list is self-describing.
mcp = FastMCP("perception")


@mcp.tool()
def ping(message: str = "hello") -> str:
    """Health check. Echoes `message` back so we can confirm the MCP round-trip.

    Args:
        message: any string to echo.
    Returns:
        A confirmation string proving the perception server is alive.
    """
    return f"perception server alive — received: {message}"


if __name__ == "__main__":
    # Default transport is stdio: the process reads MCP requests on stdin and
    # writes responses on stdout, which is what stdio_client() in the agent
    # connects to.
    mcp.run()
