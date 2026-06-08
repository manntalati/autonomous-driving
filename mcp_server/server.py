"""
Agentic Perception Platform — MCP server (Phase 0 + Phase 1).

This is the tool API surface the agent talks to.  Tools are split across
two tiers:

  Phase 0 — ping (trivial health-check, kept for smoke-testing)

  Phase 1 — full perception tools (registered via perception_tools.py):
    Core:
      list_scenes, load_frame, detect_objects, segment_scene, bev_map
    High-level driving decisions:
      check_lane_switch_safety, check_turn_clearance, check_obstacle_stop,
      check_pedestrian_crossing, estimate_following_distance, scene_summary

Each Phase-1 tool wraps the trained models behind a clean JSON API —
the agent reasons over natural-language summaries, never raw tensors.

Run standalone (quick sanity-check, no API key needed):
    python -m mcp_server.server
The stdio client in agent/mcp_client.py normally launches this for you.
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from mcp_server.perception_tools import register_all_tools

mcp = FastMCP("perception")


# ── Phase 0 ──────────────────────────────────────────────────────────────────

@mcp.tool()
def ping(message: str = "hello") -> str:
    """
    Health check. Echoes `message` back to confirm the MCP round-trip.

    Args:
        message: any string to echo.
    Returns:
        A confirmation string proving the perception server is alive.
    """
    return f"perception server alive — received: {message}"


# ── Phase 1 — register all perception tools ──────────────────────────────────

register_all_tools(mcp)


if __name__ == "__main__":
    # Default transport is stdio: the process reads MCP requests on stdin and
    # writes responses on stdout, which is what stdio_client() in the agent
    # connects to.
    mcp.run()
