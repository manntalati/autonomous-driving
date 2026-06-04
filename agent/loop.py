"""
Agentic Perception Platform — hand-built tool-use loop (Phase 0 → Phase 2).

THIS IS YOUR FILE TO IMPLEMENT. It's the "agent from scratch" piece you chose
over the Agent SDK: the loop that drives Claude through tool calls.

The contract of an Anthropic tool-use turn:
  1. You send `messages` + a `tools` schema to the Messages API.
  2. If the model wants a tool, the response has `stop_reason == "tool_use"` and
     its `.content` contains one or more `tool_use` blocks (each with .id, .name,
     .input). Any text blocks are the model "thinking out loud".
  3. You execute each tool (here: via MCPClient.call_tool), then send the results
     back as a NEW user message whose content is a list of `tool_result` blocks,
     each carrying the matching `tool_use_id`.
  4. Repeat until the model stops asking for tools (stop_reason == "end_turn"),
     then return its final text.

Phase 0 goal: prove this loop end-to-end on the trivial `ping` tool. Phase 2:
the exact same loop orchestrates the real perception tools — no rewrite needed.

Needs ANTHROPIC_API_KEY in the environment.
"""
from __future__ import annotations

from typing import Any

from anthropic import AsyncAnthropic

from agent.mcp_client import MCPClient


def mcp_tools_to_anthropic(mcp_tools: list[Any]) -> list[dict[str, Any]]:
    """Convert MCP tool defs → the Anthropic `tools` schema.

    Each MCP tool has `.name`, `.description`, and `.inputSchema` (already a JSON
    Schema dict). Anthropic wants a list of:
        {"name": ..., "description": ..., "input_schema": <that JSON Schema>}

    TODO: map each tool in `mcp_tools` to that dict and return the list.
    """
    raise NotImplementedError


async def run_agent(
    question: str,
    client: MCPClient,
    *,
    model: str = "claude-opus-4-7",
    max_turns: int = 8,
    system: str | None = None,
) -> str:
    """Drive the tool-use loop until Claude returns a final text answer.

    Args:
        question: the user's natural-language request.
        client: an already-connected MCPClient (tools come from client.list_tools()).
        model: Anthropic model id.
        max_turns: hard cap on tool-call rounds (guards against infinite loops).
        system: optional system prompt (Phase 2 puts the spatial-reasoning prompt here).

    Returns:
        The model's final assistant text.

    Implementation sketch (fill in):
        anthropic = AsyncAnthropic()
        tools = mcp_tools_to_anthropic(await client.list_tools())
        messages = [{"role": "user", "content": question}]
        for _ in range(max_turns):
            resp = await anthropic.messages.create(
                model=model, max_tokens=1024, system=system or NOT_GIVEN,
                tools=tools, messages=messages,
            )
            # append the assistant turn (resp.content) to messages
            if resp.stop_reason != "tool_use":
                return "".join(b.text for b in resp.content if b.type == "text")
            # for each tool_use block: result = await client.call_tool(b.name, b.input)
            # collect {"type": "tool_result", "tool_use_id": b.id, "content": result}
            # append them as one {"role": "user", "content": [...]} message
        raise RuntimeError("hit max_turns without a final answer")

    TODO: implement the loop above.
    """
    raise NotImplementedError
