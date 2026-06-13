"""
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
"""
from __future__ import annotations
from typing import Any, Callable
from anthropic import AsyncAnthropic
from agent.mcp_client import MCPClient
from dataclasses import dataclass, field

EventCallback = Callable[[dict], None]

@dataclass
class AgentResult:
    answer: str
    trace: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    turns: int = 0

def mcp_tools_to_anthropic(mcp_tools: list[Any]) -> list[dict[str, Any]]:
    """Convert MCP tool defs → the Anthropic `tools` schema."""
    tools = []
    for tool in mcp_tools:
        tools.append({"name": tool.name, "description": tool.description, "input_schema": tool.inputSchema})
    return tools

async def run_agent(question: str, client: MCPClient, *, model: str = "claude-opus-4-7", max_turns: int = 8, system: str | None = None, on_event: EventCallback | None = None) -> AgentResult:
    """Drive the tool-use loop until Claude returns a final text answer.

    Args:
        question: the user's natural-language request.
        client: an already-connected MCPClient (tools come from client.list_tools()).
        model: Anthropic model id.
        max_turns: hard cap on tool-call rounds (guards against infinite loops).
        system: optional system prompt (Phase 2 puts the spatial-reasoning prompt here).
        on_event: optional Callable

    Returns:
        AgentResult with answer, trace, token counts, and turn count
    """
    result = AgentResult(answer="")
    anthropic = AsyncAnthropic()
    tools = mcp_tools_to_anthropic(await client.list_tools())
    messages = [{"role": "user", "content": question}]
    for _ in range(max_turns):
        kwargs = {"model": model, "max_tokens": 1024, "tools": tools, "messages": messages}
        if system is not None:
            kwargs["system"] = system
        response = await anthropic.messages.create(**kwargs)
        result.turns += 1
        result.input_tokens += response.usage.input_tokens
        result.output_tokens += response.usage.output_tokens
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            result.answer = "".join(body.text for body in response.content if body.type == "text")
            if on_event: on_event({"type": "final", "answer": result.answer})
            return result
        tool_results = []
        for block in response.content:
            if block.type == "text":
                if on_event: on_event({"type": "text", "text": block.text})
                continue
            if block.type != "tool_use": continue
            if on_event: on_event({"type": "tool_call", "name": block.name, "input": block.input})
            result.trace.append(block.name)
            tool_output = await client.call_tool(block.name, block.input)
            if on_event: on_event({"type": "tool_result", "name": block.name, "output": tool_output})
            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": tool_output})
        messages.append({"role": "user", "content": tool_results})
    raise RuntimeError("hit max_turns without a final answer")
