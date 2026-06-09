from __future__ import annotations

import asyncio

from agent.config import load_env
from agent.loop import run_agent
from agent.mcp_client import MCPClient
from agent.prompts import SYSTEM_PROMPT

EXAMPLES = [
    # ── single-tool sanity ────────────────────────────────────────
    {"q": "How many cars are in scene-0103 frame 5?", "expects": ["load_frame", "detect_objects"]},
    {"q": "Is the road drivable in scene-1100 frame 10?", "expects": ["load_frame", "segment_scene"]},

    # ── two-tool compositions ─────────────────────────────────────
    {"q": "What's the closest car ahead in scene-0103 frame 8?",
     "expects": ["load_frame", "bev_map"]},
    {"q": "Is it safe to change into the left lane in scene-1100 frame 12?",
     "expects": ["load_frame", "check_lane_switch_safety"]},
    {"q": "Can I make a right turn safely in scene-0103 frame 15?",
     "expects": ["load_frame", "check_turn_clearance"]},

    # ── prefers decision tools over primitives ────────────────────
    {"q": "Should I stop?  scene-1100 frame 7.",
     "expects": ["load_frame", "check_obstacle_stop"]},
    {"q": "How far is the car in front of me?  scene-0103 frame 20.",
     "expects": ["load_frame", "estimate_following_distance"]},

    # ── broad question → scene_summary ────────────────────────────
    {"q": "Describe the scene at scene-1100 frame 5.",
     "expects": ["load_frame", "scene_summary"]},
]

async def run_all(client, model="claude-opus-4-7"):
    for ex in EXAMPLES:
        print(f"Q: {ex['q']}")
        answer = await run_agent(ex["q"], client, model=model, system=SYSTEM_PROMPT)
        print(f"A: {answer}\n")

async def _main() -> None:
    client = MCPClient()
    await client.connect()
    try:
        await run_all(client)
    finally:
        await client.close()

if __name__ == "__main__":
    load_env()
    asyncio.run(_main())