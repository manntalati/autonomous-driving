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
`run_agent` is one-shot: a question arrives, tools run on one frozen frame_id, an
answer returns. No notion of time, no memory between calls, no self-initiation.

This agent is the inverse — nobody asks it anything; it watches and decides when
to speak. Four added capabilities:
    1. self-initiation       — driven by Events, not questions
    2. temporal context      — knows what just happened
    3. conversational memory — does not repeat itself
    4. abstention            — declines to answer when trust is low

`run_agent` is reused for the tool-use turn (it already implements the Anthropic
protocol correctly) and now degrades instead of raising on max_turns, so one
confusing frame cannot kill a whole drive.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from agent.monitor import Event, EventType

# The autonomy-monitor persona: narrates the EGO VEHICLE's situation, like a
# Waymo rider display or Tesla FSD visualisation. Not a pedestrian assistant and
# not a general Q&A bot.
STREAMING_SYSTEM_PROMPT = """\
You are the autonomy monitor for a self-driving vehicle. You watch a continuous
stream of perception output and speak ONLY when the driver or rider needs to know
something. You never see pixels — only the JSON that perception tools return.

# What you are reacting to
Each time you are invoked, a deterministic safety monitor has already decided
something changed. You are given the triggering event, the current scene state,
and what you have recently said. Turn that into one clear sentence.

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
design domain. When `in_odd` is false you MUST NOT issue a confident advisory.
State what is degraded and what you cannot vouch for. Never smooth over low trust
with a confident-sounding answer: reporting a clear road when you cannot see is
the worst failure available to you. A driver who knows you are blind can take
over; one who trusts a false all-clear cannot.

# Invariants
- Never invent counts, distances, or classes. Only report what tools returned.
- Metres for distance, seconds for TTC, degrees for bearing.
- If a fact is unknown, say so.
"""

# Templated advisory used when the frame is outside the ODD. Deliberately NOT
# produced by the model — see the abstention note on StreamingAgent.
DEGRADED_TEMPLATE = (
    "Perception degraded — {reason}. Trust {trust:.2f}, below the {threshold:.2f} "
    "operational threshold. Do not rely on object reports until conditions improve."
)


@dataclass
class Advisory:
    frame_idx: int
    text: str
    event_type: str
    trust: float
    abstained: bool
    latency_ms: float = 0.0
    tool_calls: List[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    track_id: Optional[int] = None


class AdvisoryMemory:
    """
    What the agent has already said, so it does not repeat itself.

    Args:
        ttl_frames: how long an advisory suppresses near-duplicates.
        escalation: severity increase required to re-fire inside the TTL.

    Most suppression decisions need no model call at all, which is what separates
    a monitor from a notification spammer — and keeps API cost down.
    """

    def __init__(self, ttl_frames: int = 30, escalation: float = 0.25) -> None:
        self.ttl_frames = ttl_frames
        self.escalation = escalation
        self.history: List[Advisory] = []
        self._last: Dict[Tuple[Optional[int], str], Tuple[int, float]] = {}

    def should_suppress(self, event: Event) -> bool:
        key = (event.track_id, event.type.value)
        prev = self._last.get(key)
        if prev is None:
            return False
        last_frame, last_sev = prev
        if event.frame_idx - last_frame >= self.ttl_frames:
            return False
        # Let a materially worse situation through even inside the TTL.
        return event.severity < last_sev + self.escalation

    def record(self, advisory: Advisory, event: Event) -> None:
        self.history.append(advisory)
        self._last[(event.track_id, event.type.value)] = (event.frame_idx, event.severity)

    def recent_context(self, n: int = 3) -> str:
        """Recent advisories, formatted for the prompt so the agent reads as one observer."""
        if not self.history:
            return "(nothing said yet this drive)"
        return "\n".join(f"- frame {a.frame_idx}: {a.text}" for a in self.history[-n:])


class StreamingAgent:
    """
    Drive the loop: stream -> perception -> trust -> tracking -> events ->
    (maybe) LLM -> advisory.

    Args:
        pipeline: the Phase 7 PerceptionPipeline (or any object exposing
            `process_frame`). May be None for a dry run over pre-computed state.
        tracker / detector: the fast tier (SceneTracker, EventDetector).
        trust_scorer: Phase 11 TrustScorer; None disables abstention.
        mcp_client: connected MCPClient for the slow tier's tool calls.
        model: Anthropic model id.
        min_severity: events below this never wake the LLM.

    ABSTENTION IS A PYTHON BRANCH, NOT A PROMPT RULE (P12-4)
    --------------------------------------------------------
    A system-prompt instruction is a strong suggestion; a branch is a guarantee.
    For a safety-relevant behaviour, take the guarantee. When `in_odd` is false the
    LLM is never given the chance to make a confident object-level claim — a
    templated degraded-perception advisory is emitted instead, citing the trust
    reason. This also makes P12-5 exactly measurable: abstention correctness is a
    boolean, not a judgement about whether a sentence sounded hedged enough.

    The trust score is still passed into the prompt on the in-ODD path so the
    model can hedge proportionately at middling trust.

    COST AND LATENCY
    ----------------
    Every advisory records latency, tool-call count and tokens. Report
    events/minute and advisories/minute beside the Phase 7 FPS table:
    "17.5 FPS perception, 1 LLM call per 40 frames" is the precise systems claim
    that justifies the two-tier design.
    """

    def __init__(self, pipeline=None, tracker=None, detector=None, trust_scorer=None,
                 mcp_client=None, model: str = "claude-opus-4-7",
                 min_severity: float = 0.3, memory: Optional[AdvisoryMemory] = None) -> None:
        from agent.monitor import EventDetector, SceneTracker

        self.pipeline = pipeline
        self.tracker = tracker if tracker is not None else SceneTracker()
        self.detector = detector if detector is not None else EventDetector()
        self.trust_scorer = trust_scorer
        self.mcp_client = mcp_client
        self.model = model
        self.min_severity = min_severity
        self.memory = memory if memory is not None else AdvisoryMemory()
        self.stats = {"frames": 0, "events": 0, "llm_calls": 0, "abstentions": 0,
                      "suppressed": 0}

    def _degraded_advisory(self, event: Event, trust: Dict) -> Advisory:
        """Templated, model-free advisory for out-of-ODD frames."""
        return Advisory(
            frame_idx=event.frame_idx,
            text=DEGRADED_TEMPLATE.format(
                reason=trust.get("reason", "perception confidence low"),
                trust=float(trust.get("trust", 0.0)),
                threshold=getattr(self.trust_scorer, "odd_threshold", 0.5),
            ),
            event_type=event.type.value,
            trust=float(trust.get("trust", 0.0)),
            abstained=True,
            track_id=event.track_id,
        )

    def _build_prompt(self, event: Event, frame_state: Dict, trust: Dict) -> str:
        return (
            f"Triggering event: {event.type.value} (severity {event.severity:.2f})\n"
            f"Event details: {json.dumps(event.payload, default=float)}\n"
            f"Frame: {event.frame_idx}\n"
            f"Scene state: {json.dumps(frame_state, default=float)}\n"
            f"Perception trust: {trust.get('trust', 1.0):.2f} "
            f"(in_odd={trust.get('in_odd', True)})\n"
            f"Recently said:\n{self.memory.recent_context()}\n\n"
            f"Issue the advisory, or say nothing new is needed."
        )

    async def _handle_event(self, event: Event, frame_state: Dict, trust: Dict) -> Advisory:
        """Build the prompt, call run_agent, wrap the result in an Advisory."""
        from agent.loop import run_agent

        t0 = time.perf_counter()
        result = await run_agent(
            self._build_prompt(event, frame_state, trust),
            self.mcp_client,
            model=self.model,
            system=STREAMING_SYSTEM_PROMPT,
        )
        self.stats["llm_calls"] += 1
        return Advisory(
            frame_idx=event.frame_idx,
            text=result.answer,
            event_type=event.type.value,
            trust=float(trust.get("trust", 1.0)),
            abstained=bool(result.exhausted),   # a non-converged turn is not an assessment
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            tool_calls=list(result.trace),
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            track_id=event.track_id,
        )

    async def step(self, detections, labels, scores, timestamp_us: int, frame_idx: int,
                   ego_pose: Optional[dict] = None, trust: Optional[Dict] = None,
                   frame_state: Optional[Dict] = None,
                   on_advisory: Optional[Callable] = None) -> List[Advisory]:
        """
        Process ONE frame: fast tier always, slow tier only on salient events.

        Separated from `run` so the Streamlit demo can drive it frame by frame and
        `streaming_eval` can replay pre-computed perception without a pipeline.
        """
        self.stats["frames"] += 1
        tracks = self.tracker.update(detections, labels, scores, timestamp_us,
                                     frame_idx, ego_pose=ego_pose)
        events = self.detector.step(tracks, frame_idx, trust=trust)
        self.stats["events"] += len(events)

        out: List[Advisory] = []
        for event in events:
            if event.severity < self.min_severity:
                continue
            if self.memory.should_suppress(event):
                self.stats["suppressed"] += 1
                continue

            t = trust or {"trust": 1.0, "in_odd": True}
            if not t.get("in_odd", True):
                advisory = self._degraded_advisory(event, t)
                self.stats["abstentions"] += 1
            elif self.mcp_client is None:
                # No LLM available (dry run / eval): emit a deterministic
                # description so the fast tier can still be evaluated end to end.
                advisory = Advisory(
                    frame_idx=event.frame_idx,
                    text=f"[{event.type.value}] {json.dumps(event.payload, default=float)}",
                    event_type=event.type.value, trust=float(t.get("trust", 1.0)),
                    abstained=False, track_id=event.track_id,
                )
            else:
                advisory = await self._handle_event(event, frame_state or {}, t)

            self.memory.record(advisory, event)
            out.append(advisory)
            if on_advisory:
                on_advisory(advisory)
        return out

    async def run(self, stream, on_advisory: Optional[Callable] = None) -> List[Advisory]:
        """
        Consume a FrameStream to completion.

        Requires a pipeline exposing `process_frame(image) -> dict` with BEV
        detections. `on_advisory` is a callback for live UI updates.
        """
        if self.pipeline is None:
            raise RuntimeError("run() needs a pipeline; use step() to drive frames manually")

        advisories: List[Advisory] = []
        for frame in stream:
            result = self.pipeline.process_frame(frame.image)
            det = result.get("bev_boxes", [])
            labels = result.get("bev_labels", [])
            scores = result.get("bev_scores", [])

            trust = None
            if self.trust_scorer is not None and "features" in result:
                trust = self.trust_scorer.score_frame(det, scores, labels, result["features"])

            advisories += await self.step(
                det, labels, scores, frame.timestamp_us, frame.frame_idx,
                ego_pose=frame.ego_pose, trust=trust,
                frame_state=result.get("summary"), on_advisory=on_advisory,
            )
        return advisories

    def rate_summary(self, stream_duration_s: float) -> Dict[str, float]:
        """Events and advisories per minute — the two-tier cost justification."""
        mins = max(stream_duration_s / 60.0, 1e-9)
        return {
            "frames": self.stats["frames"],
            "events_per_min": self.stats["events"] / mins,
            "advisories_per_min": len(self.memory.history) / mins,
            "llm_calls": self.stats["llm_calls"],
            "llm_calls_per_frame": self.stats["llm_calls"] / max(self.stats["frames"], 1),
            "abstentions": self.stats["abstentions"],
            "suppressed": self.stats["suppressed"],
        }
