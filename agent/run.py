"""
Agentic Perception Platform — CLI entry point (Phase 0).

Wires the pieces together: connect to the MCP server, run the agent loop on a
question, print the answer. Provided complete so you can smoke-test your
agent/loop.py implementation immediately.

Usage:
    export ANTHROPIC_API_KEY=...
    python -m agent.run "ping the perception server with the word banana"
"""
from __future__ import annotations

import argparse
import asyncio

from agent.config import load_env
from agent.loop import run_agent
from agent.mcp_client import MCPClient


async def _main(question: str, model: str) -> None:
    load_env()  # pulls ANTHROPIC_API_KEY from .env if not already exported
    client = MCPClient()
    await client.connect()
    try:
        answer = await run_agent(question, client, model=model)
        print(answer)
    finally:
        await client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agentic Perception Platform CLI")
    parser.add_argument("question", help="natural-language request for the agent")
    parser.add_argument("--model", default="claude-opus-4-7")
    args = parser.parse_args()
    asyncio.run(_main(args.question, args.model))
