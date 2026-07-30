"""
P12-2 — Fast tier: per-frame state tracking and deterministic event detection.

NO LLM RUNS IN THIS FILE. That is the entire design point.

WHY TWO TIERS
-------------
The naive "agentic" design calls an LLM every frame. At 12 Hz that is ~1,900 API
calls per scene, seconds of latency per frame, and an agent that narrates "car
ahead" 1,900 times. It is also not what production systems do: Waymo and Tesla
run cheap deterministic safety monitors at high rate and reserve expensive
reasoning for when something actually warrants it.

So:
    FAST TIER (here)  — every frame, pure Python/tensor ops, sub-millisecond.
        Tracks object state, computes kinematics, raises typed Events.
    SLOW TIER (streaming_agent.py) — LLM, only when an Event fires.

The hard and interesting problem in this file is SALIENCE: deciding what is worth
speaking about. A system that knows when to stay quiet is much harder to build
than one that talks constantly, and it is the part a reviewer will find most
credible. Most frames should produce zero events.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class EventType(str, Enum):
    """Typed events the fast tier can raise."""
    OBJECT_ENTERED_PATH = "object_entered_path"      # something moved into the ego lane
    TTC_BELOW_THRESHOLD = "ttc_below_threshold"      # closing too fast
    PEDESTRIAN_APPROACHING_ROAD = "ped_approaching_road"
    OBJECT_LOST = "object_lost"                      # tracked object vanished unexpectedly
    TRUST_DEGRADED = "trust_degraded"                # Phase 11 trust crossed the ODD bound
    TRUST_RECOVERED = "trust_recovered"


@dataclass
class Event:
    type: EventType
    frame_idx: int
    severity: float                  # 0-1, drives whether the slow tier is woken
    track_id: Optional[int] = None
    payload: Dict = field(default_factory=dict)   # ttc_s, range_m, class, etc.


@dataclass
class TrackState:
    """Per-object state carried across frames."""
    track_id: int
    label: int
    history: List[Tuple[int, float, float]]   # (timestamp_us, x, y) in ego frame
    last_seen_frame: int
    missed_frames: int = 0
    announced: bool = False           # has the slow tier already spoken about this?


class SceneTracker:
    """
    Maintain object identity across frames from detections alone (no GT).

    Args:
        max_missed: frames an unseen track survives before deletion.
        gate_m: maximum association distance in metres.

    ASSOCIATION
    -----------
    Greedy nearest-centre matching with a distance gate is sufficient here, and
    it matches the matcher chosen for P10-4 so behaviour is consistent across the
    project. Predict each track's expected position with a constant-velocity step
    before matching, or fast-moving objects will fail to associate at 2 Hz — the
    gate has to cover the real displacement, and at 2 Hz a 15 m/s vehicle moves
    7.5 m between keyframes.

    Do not reach for nuScenes `instance_token`. `evaluation/eval_flicker.py` uses
    it legitimately because it is an offline evaluation, but this runs at
    inference where no labels exist. Using GT identity here would make the whole
    demo a lie.

    EGO MOTION
    ----------
    Detections are in the EGO frame, which moves with the vehicle. A perfectly
    stationary parked car has a changing ego-frame position, so naive velocity
    estimates will show every static object drifting toward you at the ego speed.
    Either transform tracks to the global frame using `ego_pose` before computing
    velocity, or subtract the ego motion explicitly. Getting this wrong makes
    every parked car look like an imminent collision and will flood the agent
    with false TTC events — it is the most likely single bug in this phase.
    """

    def __init__(self, max_missed: int = 3, gate_m: float = 4.0) -> None:
        raise NotImplementedError("P12-2")

    def update(self, detections, labels, scores, timestamp_us: int, frame_idx: int,
               ego_pose: Optional[dict] = None) -> Dict[int, TrackState]:
        """Associate detections to tracks; returns the live track table."""
        raise NotImplementedError("P12-2")


class EventDetector:
    """
    Turn tracked state into salient Events.

    Args:
        ttc_threshold_s: time-to-collision below which to raise an event.
        path_half_width_m: half-width of the ego corridor (2 m matches the
            "current lane |y| <= 2 m" convention already in agent/prompts.py —
            keep them consistent or the agent's language will contradict its
            own triggers).
        cooldown_frames: minimum frames between repeat events of the same type
            for the same track.

    SALIENCE RULES — the heart of P12
    ---------------------------------
    Fire on CHANGE and on RISK, never on mere presence:
        * a car 30 m ahead in a steady state       -> no event, ever
        * that car crossing into the ego corridor  -> OBJECT_ENTERED_PATH
        * TTC falling below threshold              -> TTC_BELOW_THRESHOLD
        * a pedestrian's lateral velocity pointing at the roadway
                                                    -> PEDESTRIAN_APPROACHING_ROAD
        * a confidently-tracked object disappearing -> OBJECT_LOST
        * trust crossing the ODD boundary          -> TRUST_DEGRADED

    HYSTERESIS IS MANDATORY. A raw threshold on a noisy signal produces
    chattering: TTC oscillating around 4.0 s fires an event every other frame.
    Use separate enter/exit thresholds (e.g. fire below 3.5 s, clear above 4.5 s)
    plus `cooldown_frames`. Without this the demo will be unwatchable and the
    false-alarm rate in P12-5 will be dominated by chatter rather than by real
    detector errors — which would make the metric meaningless.

    TTC DEFINITION — state it explicitly
        ttc = range / closing_speed, where closing_speed is the component of
        relative velocity along the line joining ego and object. Guard the
        divide: closing_speed <= 0 means the object is receding, so TTC is
        infinite, not negative. An unguarded divide here yields a large negative
        TTC that trivially passes a `< threshold` test and fires a collision
        warning for every car driving away from you.
    """

    def __init__(self, ttc_threshold_s: float = 4.0, path_half_width_m: float = 2.0,
                 cooldown_frames: int = 10) -> None:
        raise NotImplementedError("P12-2")

    def step(self, tracks: Dict[int, TrackState], frame_idx: int,
             trust: Optional[Dict] = None) -> List[Event]:
        """
        Evaluate all rules for this frame.

        Returns: list of Events, usually empty. If most frames yield events, the
        thresholds are wrong — measure the event rate per scene and tune until
        it matches what a human would actually remark on during that drive.
        """
        raise NotImplementedError("P12-2")
