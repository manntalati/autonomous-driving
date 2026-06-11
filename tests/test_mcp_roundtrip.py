"""
Phase 0 — MCP round-trip smoke test.

Launches the perception MCP server over stdio, lists its tools, and calls
`ping`. No Anthropic API key required (this exercises the MCP layer only, not
the agent loop). Uses asyncio.run() so we don't need pytest-asyncio.
"""
import asyncio

import pytest

# Skip cleanly if the optional `mcp` dependency isn't installed yet.
pytest.importorskip("mcp")

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
