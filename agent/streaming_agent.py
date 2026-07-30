"""
P12-3 / P12-4 — Slow tier: event-triggered LLM advisories with abstention.

    FrameStream ──> PerceptionPipeline ──> TrustScorer ──> SceneTracker
                                                                │
                                                          EventDetector
                                                                │
                                                       (only if an event fires)
                                                                ▼
                                                        StreamingAgent ──> advisory

WHAT MAKES THIS DIFFERENT FROM agent/loop.py
--------------------------------------------
`run_agent` is one-shot: a question arrives, tools are called on one frozen
frame_id, an answer is returned, the process ends. It has no notion of time, no
memory between calls, and no way to initiate anything.

This agent is the inverse: nobody asks it anything. It watches, and it decides
when to speak. Four capabilities have to be added:

    1. self-initiation      — driven by Events, not user questions
    2. temporal context     — it knows what just happened, not just what is
    3. conversational memory— it does not repeat itself
    4. abstention           — it can decline to answer when trust is low

Reuse `run_agent` for the actual tool-use turn. It already handles the Anthropic
tool-use protocol correctly; wrap it rather than reimplementing it.

ONE BUG TO FIX IN THE WRAPPED CODE
----------------------------------
`agent/loop.py` raises RuntimeError when it hits `max_turns`. That is fine for a
CLI one-shot and fatal for a monitor that must survive a whole drive. Catch it,
emit a degraded advisory ("unable to assess — reasoning did not converge"), and
continue the stream. A perception monitor that crashes mid-drive because one
frame confused the model is worse than one that admits a gap.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from agent.monitor import Event, EventType

# The autonomy-monitor persona: this agent narrates the EGO VEHICLE's situation,
# the way Waymo's rider display or Tesla FSD's visualisation does. It is not a
# pedestrian assistant and not a general Q&A bot.
STREAMING_SYSTEM_PROMPT = """\
You are the autonomy monitor for a self-driving vehicle. You watch a continuous
stream of perception output and speak ONLY when the driver or rider needs to know
something. You never see pixels — only the JSON that perception tools return.

# What you are reacting to
Each time you are invoked, a deterministic safety monitor has already decided
something changed. You are given the triggering event, the current scene state,
and what you have recently said. Your job is to turn that into one clear sentence.

# Style
- ONE sentence. Two only if a safety action is implied.
- Lead with the action or risk, then the grounding fact.
    Good: "Slowing — pedestrian entering the crosswalk 12 m ahead."
    Bad:  "I detected a pedestrian using the detect_objects tool."
- Always cite a concrete number: range in metres, TTC in seconds, or bearing.
- Never repeat an advisory you have already given for the same object unless the
  situation has materially changed.

# Trust and abstention  (THE MOST IMPORTANT RULE)
You are told a trust score and whether perception is inside its operational
design domain. When `in_odd` is false, you MUST NOT issue a confident advisory.
Say what is degraded and what you cannot vouch for:
    "Perception degraded — low light, 6 of 8 detections lack radar
     corroboration. Do not rely on object reports until conditions improve."
Never smooth over low trust with a confident-sounding answer. Reporting a clear
road when you cannot see is the single worst failure available to you — a driver
who knows you are blind can take over; one who trusts a false all-clear cannot.

# Invariants
- Never invent counts, distances, or classes. Only report what tools returned.
- Metres for distance, seconds for TTC, degrees for bearing.
- If a fact is unknown, say so.
"""


@dataclass
class Advisory:
    frame_idx: int
    text: str
    event: Event
    trust: float
    abstained: bool
    latency_ms: float = 0.0
    tool_calls: List[str] = field(default_factory=list)


class AdvisoryMemory:
    """
    What the agent has already said, so it does not repeat itself.

    Args:
        ttl_frames: how long an advisory suppresses near-duplicates.

    Keep a compact rolling record: (track_id, EventType, frame_idx, text). Before
    invoking the LLM, check whether this track already has a live advisory of this
    type; if so, suppress unless severity has risen materially.

    This is what separates a monitor from a notification spammer, and it is
    cheaper than it looks: most suppression decisions need no model call at all,
    which also keeps API cost down.

    Feed a short window of recent advisories into the prompt as context so the
    model can write "still braking — that pedestrian is now in the lane" rather
    than restating the situation from scratch. Continuity is most of what makes
    it read as a single coherent observer instead of a series of disconnected
    alerts.
    """

    def __init__(self, ttl_frames: int = 30) -> None:
        raise NotImplementedError("P12-3")

    def should_suppress(self, event: Event) -> bool:
        raise NotImplementedError("P12-3")

    def record(self, advisory: Advisory) -> None:
        raise NotImplementedError("P12-3")

    def recent_context(self, n: int = 3) -> str:
        """Recent advisories, formatted for the prompt."""
        raise NotImplementedError("P12-3")


class StreamingAgent:
    """
    Drive the whole loop: stream -> perception -> trust -> tracking -> events ->
    (maybe) LLM -> advisory.

    Args:
        pipeline: the Phase 7 PerceptionPipeline.
        trust_scorer: the Phase 11 TrustScorer (None disables abstention).
        mcp_client: connected MCPClient for the slow tier's tool calls.
        model: Anthropic model id.
        min_severity: events below this never wake the LLM.

    ABSTENTION LOGIC (P12-4) — decide in code, not in the prompt
    -----------------------------------------------------------
    Do not rely on the model to police itself. Gate structurally:

        trust = trust_scorer.score_frame(...)
        if not trust["in_odd"]:
            emit a templated degraded-perception advisory citing trust["reason"]
            do NOT call the LLM with object-level claims it might present
            confidently
        else:
            invoke the LLM with the event, scene state, and trust context

    An instruction in a system prompt is a strong suggestion; a branch in Python
    is a guarantee. For a safety-relevant behaviour, take the guarantee. It is
    also far easier to evaluate — P12-5 can check abstention correctness exactly,
    with no judgement call about whether a sentence "sounded" hedged enough.

    Do still pass the trust score into the prompt on the in-ODD path, so the model
    can hedge proportionately at middling trust.

    COST AND LATENCY — measure and report
        Log per advisory: wall-clock latency, tool-call count, tokens.
        Report events/minute and advisories/minute alongside the Phase 7 FPS
        table. "17.5 FPS perception, 1 LLM call per 40 frames" is a precise and
        credible systems claim, and it is the number that justifies the two-tier
        design.
    """

    def __init__(self, pipeline, trust_scorer=None, mcp_client=None,
                 model: str = "claude-opus-4-7", min_severity: float = 0.3) -> None:
        raise NotImplementedError("P12-3")

    async def run(self, stream, on_advisory=None) -> List[Advisory]:
        """
        Consume a FrameStream to completion, yielding advisories as they occur.

        `on_advisory` is a callback for live UI updates (the Streamlit panel).
        Return the full advisory list for offline evaluation.
        """
        raise NotImplementedError("P12-3")

    async def _handle_event(self, event: Event, frame_state: Dict, trust: Dict) -> Advisory:
        """Build the prompt, call run_agent, wrap the result in an Advisory."""
        raise NotImplementedError("P12-3")
