"""
Phase 0 — MCP round-trip smoke test.

Launches the perception MCP server over stdio, lists its tools, and calls
`ping`. No Anthropic API key required (this exercises the MCP layer only, not
the agent loop). Uses asyncio.run() so we don't need pytest-asyncio.
"""
import asyncio

import pytest

# Skip cleanly if the optional MCP *server* stack isn't available.
#
# This guarded on plain `mcp` before, which is the wrong check: the client half
# imports fine while the server subprocess is what needs FastMCP. When mcp 2.0
# removed `mcp.server.fastmcp` the guard passed, the server died on import, and
# the test failed with an opaque "Connection closed" instead of skipping.
# Guard on the module the server actually imports.
pytest.importorskip("mcp.server.fastmcp",
                    reason="mcp.server.fastmcp unavailable (removed in mcp 2.0) "
                           "— pin mcp<2.0 to run the MCP round-trip test")

from agent.mcp_client import MCPClient


async def _roundtrip() -> tuple[list[str], str]:
    client = MCPClient()
    await client.connect()
    try:
        tools = await client.list_tools()
        names = [t.name for t in tools]
        echoed = await client.call_tool("ping", {"message": "banana"})
        return names, echoed
    finally:
        await client.close()


def test_ping_roundtrip():
    names, echoed = asyncio.run(_roundtrip())
    assert "ping" in names
    assert "banana" in echoed
