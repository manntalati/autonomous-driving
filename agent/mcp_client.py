"""
Agentic Perception Platform — MCP client plumbing (Phase 0).

Thin wrapper around the MCP Python SDK's stdio client. It launches the
perception MCP server as a subprocess, holds the session open, and exposes the
three calls the agent loop needs: list_tools / call_tool / close.

This file is pure SDK plumbing (no agent reasoning lives here) so it's provided
complete — the from-scratch logic you implement is in agent/loop.py.
"""
from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClient:
    """Owns one stdio connection to the perception MCP server."""

    def __init__(self) -> None:
        self.session: ClientSession | None = None
        self._stack = AsyncExitStack()

    async def connect(self, server_module: str = "mcp_server.server") -> None:
        """Launch the server (`python -m <server_module>`) and initialize a session.

        We enter the stdio transport and the ClientSession into a single
        AsyncExitStack so a later close() tears both down in order.
        """
        params = StdioServerParameters(command="python", args=["-m", server_module])
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self.session = await self._stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()

    async def list_tools(self) -> list[Any]:
        """Return the server's tool definitions (name / description / inputSchema)."""
        assert self.session is not None, "call connect() first"
        resp = await self.session.list_tools()
        return resp.tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Invoke a tool and return its text content as a plain string.

        MCP returns a list of typed content blocks; for Phase 0/1 our tools emit
        a single text (JSON) block, so we concatenate any text parts.
        """
        assert self.session is not None, "call connect() first"
        result = await self.session.call_tool(name, arguments)
        return "".join(getattr(b, "text", "") for b in result.content)

    async def close(self) -> None:
        """Tear down the session and the server subprocess."""
        await self._stack.aclose()
        self.session = None
