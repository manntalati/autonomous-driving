"""
P12-2 — Fast tier: per-frame state tracking and deterministic event detection.

NO LLM RUNS IN THIS FILE. That is the design point.

WHY TWO TIERS
-------------
The naive "agentic" design calls an LLM every frame. At 12 Hz that is ~1,900 API
calls per scene, seconds of latency per frame, and an agent that narrates "car
ahead" 1,900 times. It is also not what production systems do: Waymo and Tesla run
cheap deterministic safety monitors at high rate and reserve expensive reasoning
for when something warrants it.

    FAST TIER (here) — every frame, pure numpy, sub-millisecond. Tracks object
        state, computes kinematics, raises typed Events.
    SLOW TIER (streaming_agent.py) — LLM, only when an Event fires.

The hard problem here is SALIENCE: deciding what is worth speaking about. Most
frames must produce zero events. A system that knows when to stay quiet is much
harder than one that talks constantly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.geometry import ego_to_global, global_to_ego, time_to_collision

# Matches the "current lane |y| <= 2 m" convention in agent/prompts.py. Keep them
# equal or the agent's language will contradict its own triggers.
EGO_HALF_WIDTH_M = 2.0


class EventType(str, Enum):
    OBJECT_ENTERED_PATH = "object_entered_path"
    TTC_BELOW_THRESHOLD = "ttc_below_threshold"
    PEDESTRIAN_APPROACHING_ROAD = "ped_approaching_road"
    OBJECT_LOST = "object_lost"
    TRUST_DEGRADED = "trust_degraded"
    TRUST_RECOVERED = "trust_recovered"


@dataclass
class Event:
    type: EventType
    frame_idx: int
    severity: float                       # 0-1; gates whether the LLM is woken
    track_id: Optional[int] = None
    payload: Dict = field(default_factory=dict)


@dataclass
class TrackState:
    """Per-object state carried across frames."""
    track_id: int
    label: int
    xy: np.ndarray                        # latest ego-frame position
    xy_global: np.ndarray                 # latest global position
    score: float
    timestamp_us: int
    last_seen_frame: int
    missed_frames: int = 0
    history: List[Tuple[int, float, float]] = field(default_factory=list)  # (t_us, gx, gy)
    # Absolute velocity of the object (ego-motion removed), in ego axes. Zero for
    # a genuinely parked car. Use this to decide whether an object is MOVING.
    velocity_ego: np.ndarray = field(default_factory=lambda: np.zeros(2))
    # Velocity of the object RELATIVE to the ego vehicle, in ego axes. This is
    # what determines collision, and is what TTC must use — a parked car with
    # absolute velocity 0 is still on a collision course if the ego drives at it.
    relative_velocity_ego: np.ndarray = field(default_factory=lambda: np.zeros(2))
    in_path: bool = False
    was_in_path: bool = False
    announced: bool = False


class SceneTracker:
    """
    Maintain object identity across frames from detections alone (no GT).

    Args:
        max_missed: frames an unseen track survives before deletion.
        gate_m: maximum association distance in metres.

    ASSOCIATION
    -----------
    Greedy nearest-centre matching with a distance gate, matching the P10-4
    matcher so behaviour is consistent project-wide. Association happens in the
    GLOBAL frame after ego-motion compensation, which is what makes a fixed gate
    meaningful — in the ego frame a parked car moves by the ego displacement
    (7.5 m between keyframes at 15 m/s), which would blow any sane gate.

    Ground-truth `instance_token` is deliberately not used. eval_flicker.py uses it
    legitimately as an offline evaluation; here we are at inference, where no
    labels exist, and using them would make the demo a lie.

    EGO MOTION
    ----------
    Velocity is computed from GLOBAL positions and then rotated back into the ego
    frame. Differencing ego-frame positions instead would measure ego motion, and
    every parked car would read as closing at the ego speed — flooding the agent
    with false collision warnings. See utils/geometry.py.
    """

    def __init__(self, max_missed: int = 3, gate_m: float = 4.0) -> None:
        self.max_missed = max_missed
        self.gate_m = gate_m
        self.tracks: Dict[int, TrackState] = {}
        self._next_id = 0
        # Ego state, needed to convert absolute object velocity into the relative
        # velocity that TTC requires.
        self._prev_ego: Optional[np.ndarray] = None
        self._prev_ego_t: Optional[int] = None
        self.ego_velocity_ego_frame: np.ndarray = np.zeros(2)

    def update(self, detections, labels, scores, timestamp_us: int, frame_idx: int,
               ego_pose: Optional[dict] = None) -> Dict[int, TrackState]:
        """
        Associate detections to tracks; returns the live track table.

        Args:
            detections: (M, >=2) ego-frame BEV boxes ([x, y, ...]).
            labels: (M,) class ids.
            scores: (M,) confidences.
            timestamp_us: nuScenes microsecond timestamp.
            frame_idx: index in the stream.
            ego_pose: this frame's ego_pose record. Required for ego-motion
                compensation; without it positions are treated as already global.
        """
        det = np.asarray(detections, dtype=np.float64).reshape(-1, np.shape(detections)[-1] if len(detections) else 2)
        xy = det[:, :2] if len(det) else np.zeros((0, 2))
        labels = np.asarray(labels).reshape(-1).astype(int)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)

        xy_g = ego_to_global(xy, ego_pose) if (ego_pose is not None and len(xy)) else xy
        self._update_ego_velocity(ego_pose, timestamp_us)

        unmatched = set(range(len(xy)))
        for t in self.tracks.values():
            t.missed_frames += 1

        # greedy nearest-centre in the global frame
        if len(xy_g) and self.tracks:
            tids = list(self.tracks)
            tpos = np.array([self.tracks[i].xy_global for i in tids])
            dist = np.sqrt(((xy_g[:, None, :] - tpos[None, :, :]) ** 2).sum(-1))
            used_t = set()
            order = np.dstack(np.unravel_index(np.argsort(dist, axis=None), dist.shape))[0]
            for di, ti in order:
                if dist[di, ti] > self.gate_m:
                    break
                if di not in unmatched or tids[ti] in used_t:
                    continue
                if self.tracks[tids[ti]].label != labels[di]:
                    continue
                self._update_track(self.tracks[tids[ti]], xy[di], xy_g[di], labels[di],
                                   scores[di], timestamp_us, frame_idx, ego_pose)
                unmatched.discard(di)
                used_t.add(tids[ti])

        for di in sorted(unmatched):
            self._new_track(xy[di], xy_g[di], labels[di], scores[di], timestamp_us, frame_idx)

        for tid in [i for i, t in self.tracks.items() if t.missed_frames > self.max_missed]:
            del self.tracks[tid]
        return self.tracks

    def _update_ego_velocity(self, ego_pose: Optional[dict], t_us: int) -> None:
        """Ego velocity in the CURRENT ego frame, from consecutive ego_pose records."""
        if ego_pose is None:
            self.ego_velocity_ego_frame = np.zeros(2)
            return
        pos = np.asarray(ego_pose["translation"], dtype=np.float64)[:2]
        if self._prev_ego is not None and self._prev_ego_t is not None:
            dt = (t_us - self._prev_ego_t) / 1e6
            if dt > 0:
                from pyquaternion import Quaternion
                v_global = (pos - self._prev_ego) / dt
                R = Quaternion(ego_pose["rotation"]).rotation_matrix[:2, :2]
                self.ego_velocity_ego_frame = v_global @ R
        self._prev_ego, self._prev_ego_t = pos, int(t_us)

    def _new_track(self, xy, xy_g, label, score, t_us, frame_idx) -> None:
        tid = self._next_id
        self._next_id += 1
        self.tracks[tid] = TrackState(
            track_id=tid, label=int(label), xy=np.asarray(xy, dtype=np.float64),
            xy_global=np.asarray(xy_g, dtype=np.float64), score=float(score),
            timestamp_us=int(t_us), last_seen_frame=frame_idx, missed_frames=0,
            history=[(int(t_us), float(xy_g[0]), float(xy_g[1]))],
            in_path=abs(float(xy[1])) <= EGO_HALF_WIDTH_M,
            was_in_path=abs(float(xy[1])) <= EGO_HALF_WIDTH_M,
        )

    def _update_track(self, t: TrackState, xy, xy_g, label, score, t_us, frame_idx, ego_pose) -> None:
        dt = (t_us - t.timestamp_us) / 1e6
        if dt > 0:
            v_global = (np.asarray(xy_g) - t.xy_global) / dt
            if ego_pose is not None:
                from pyquaternion import Quaternion
                R = Quaternion(ego_pose["rotation"]).rotation_matrix[:2, :2]
                t.velocity_ego = v_global @ R          # global -> ego rotation
            else:
                t.velocity_ego = v_global
            # Relative velocity is what collides. A parked car has absolute
            # velocity 0 but closes on the ego at the ego's own speed.
            t.relative_velocity_ego = t.velocity_ego - self.ego_velocity_ego_frame
        t.was_in_path = t.in_path
        t.xy = np.asarray(xy, dtype=np.float64)
        t.xy_global = np.asarray(xy_g, dtype=np.float64)
        t.label = int(label)
        t.score = float(score)
        t.timestamp_us = int(t_us)
        t.last_seen_frame = frame_idx
        t.missed_frames = 0
        t.in_path = abs(float(xy[1])) <= EGO_HALF_WIDTH_M
        t.history.append((int(t_us), float(xy_g[0]), float(xy_g[1])))
        if len(t.history) > 20:
            t.history.pop(0)

    def history_for_signals(self, window: int = 5) -> Dict[int, List[Optional[dict]]]:
        """Shape track state for models.uncertainty.signals.temporal_instability."""
        out: Dict[int, List[Optional[dict]]] = {}
        for tid, t in self.tracks.items():
            out[tid] = [{"score": t.score, "xy": (t.xy_global[0], t.xy_global[1])}
                        for _ in range(min(len(t.history), window))]
        return out


class EventDetector:
    """
    Turn tracked state into salient Events.

    Args:
        ttc_enter_s / ttc_exit_s: hysteresis band for the TTC rule.
        path_half_width_m: half-width of the ego corridor.
        cooldown_frames: minimum frames between repeat events of the same type
            for the same track.
        ped_lateral_speed: m/s of lateral closing that counts as approaching.

    SALIENCE — fire on CHANGE and RISK, never on presence
    -----------------------------------------------------
        car 30 m ahead, steady            -> no event, ever
        that car crossing into the path   -> OBJECT_ENTERED_PATH
        TTC falling below threshold       -> TTC_BELOW_THRESHOLD
        pedestrian moving toward the road -> PEDESTRIAN_APPROACHING_ROAD
        confident track vanishing         -> OBJECT_LOST
        trust crossing the ODD boundary   -> TRUST_DEGRADED / TRUST_RECOVERED

    HYSTERESIS IS MANDATORY. A raw threshold on a noisy signal chatters: TTC
    oscillating around 4.0 s would fire every other frame. Separate enter/exit
    thresholds plus a cooldown prevent that. Without it the demo is unwatchable
    and P12-5's false-alarm rate measures chatter rather than detector error.
    """

    def __init__(self, ttc_enter_s: float = 3.5, ttc_exit_s: float = 4.5,
                 path_half_width_m: float = EGO_HALF_WIDTH_M,
                 cooldown_frames: int = 10, ped_lateral_speed: float = 1.0,
                 ttc_corridor_m: float = 3.0, global_cooldown_frames: int = 4,
                 relevance_range_m: float = 25.0, ped_entry_horizon_s: float = 4.0) -> None:
        self.ttc_enter_s = ttc_enter_s
        self.ttc_exit_s = ttc_exit_s
        self.path_half_width_m = path_half_width_m
        self.ttc_corridor_m = ttc_corridor_m
        self.cooldown_frames = cooldown_frames
        self.ped_lateral_speed = ped_lateral_speed
        self.global_cooldown_frames = global_cooldown_frames
        self.relevance_range_m = relevance_range_m
        self.ped_entry_horizon_s = ped_entry_horizon_s
        self._last_fired: Dict[Tuple[int, EventType], int] = {}
        self._last_any_frame: Optional[int] = None
        self._ttc_active: set = set()
        self._trust_degraded = False

    def _cool(self, tid: Optional[int], etype: EventType, frame_idx: int) -> bool:
        """True if this (track, type) is still in cooldown."""
        key = (tid if tid is not None else -1, etype)
        last = self._last_fired.get(key)
        if last is not None and frame_idx - last < self.cooldown_frames:
            return True
        self._last_fired[key] = frame_idx
        return False

    def step(self, tracks: Dict[int, TrackState], frame_idx: int,
             trust: Optional[Dict] = None) -> List[Event]:
        """
        Evaluate all rules for this frame. Usually returns an empty list; if most
        frames produce events, the thresholds are wrong.
        """
        events: List[Event] = []

        # Trust / ODD boundary — frame-level, no track.
        if trust is not None:
            if not trust.get("in_odd", True) and not self._trust_degraded:
                self._trust_degraded = True
                if not self._cool(None, EventType.TRUST_DEGRADED, frame_idx):
                    events.append(Event(EventType.TRUST_DEGRADED, frame_idx,
                                        severity=1.0 - float(trust.get("trust", 0.0)),
                                        payload={"trust": trust.get("trust"),
                                                 "reason": trust.get("reason")}))
            elif trust.get("in_odd", True) and self._trust_degraded:
                self._trust_degraded = False
                if not self._cool(None, EventType.TRUST_RECOVERED, frame_idx):
                    events.append(Event(EventType.TRUST_RECOVERED, frame_idx, severity=0.3,
                                        payload={"trust": trust.get("trust")}))

        for tid, t in tracks.items():
            if t.missed_frames > 0:
                # Only flag a *confident, established* track vanishing; new or weak
                # tracks blink constantly and would dominate the event stream.
                # ...and only if it was RELEVANT. Measured on real scenes, an
                # unrestricted rule produced 73 of 136 events, mostly cars fading
                # out at 40+ m — which no human would remark on. Require the lost
                # object to have been close or in the ego path.
                lost_range = float(np.hypot(t.xy[0], t.xy[1]))
                relevant = t.was_in_path or lost_range <= self.relevance_range_m
                if (t.missed_frames == 1 and t.score > 0.5 and len(t.history) >= 3
                        and relevant
                        and not self._cool(tid, EventType.OBJECT_LOST, frame_idx)):
                    events.append(Event(EventType.OBJECT_LOST, frame_idx,
                                        severity=float(np.clip(1.0 - lost_range / 40.0, 0.1, 1.0)),
                                        track_id=tid,
                                        payload={"label": t.label, "range_m": lost_range}))
                continue

            rng = float(np.hypot(t.xy[0], t.xy[1]))

            if t.in_path and not t.was_in_path and not self._cool(tid, EventType.OBJECT_ENTERED_PATH, frame_idx):
                events.append(Event(EventType.OBJECT_ENTERED_PATH, frame_idx,
                                    severity=float(np.clip(1.0 - rng / 50.0, 0.1, 1.0)),
                                    track_id=tid,
                                    payload={"label": t.label, "range_m": rng,
                                             "lateral_m": float(t.xy[1])}))

            # TTC is only meaningful for objects the ego could actually hit.
            # Line-of-sight closing speed is non-zero for anything you merely
            # drive PAST — a parked car 8 m to the side produces a shrinking range
            # and would fire a spurious collision warning. Real AEB systems gate
            # on an in-path assessment for exactly this reason. The corridor here
            # is slightly wider than the "entered path" corridor so an object
            # straddling the boundary still warns.
            ttc = (time_to_collision(t.xy, t.relative_velocity_ego)
                   if abs(float(t.xy[1])) <= self.ttc_corridor_m else float("inf"))
            if np.isfinite(ttc):
                if ttc < self.ttc_enter_s and tid not in self._ttc_active:
                    self._ttc_active.add(tid)
                    if not self._cool(tid, EventType.TTC_BELOW_THRESHOLD, frame_idx):
                        events.append(Event(EventType.TTC_BELOW_THRESHOLD, frame_idx,
                                            severity=float(np.clip(1.0 - ttc / self.ttc_enter_s, 0.1, 1.0)),
                                            track_id=tid,
                                            payload={"label": t.label, "ttc_s": float(ttc),
                                                     "range_m": rng}))
                elif ttc > self.ttc_exit_s:
                    self._ttc_active.discard(tid)   # hysteresis: clear only well above
            else:
                self._ttc_active.discard(tid)

            # Pedestrian (class 1) or cyclist (2) moving laterally toward the path.
            if t.label in (1, 2) and not t.in_path and rng <= self.relevance_range_m:
                closing_lateral = -np.sign(t.xy[1]) * t.velocity_ego[1]
                # Require not just lateral motion but motion that will actually
                # put them in the ego path soon. Ordinary walking is ~1.4 m/s, so
                # a bare speed threshold fires for every pedestrian on a footpath
                # who drifts slightly kerbward.
                gap = max(abs(float(t.xy[1])) - self.path_half_width_m, 0.0)
                entry_s = gap / closing_lateral if closing_lateral > 1e-6 else float("inf")
                if (closing_lateral > self.ped_lateral_speed
                        and entry_s <= self.ped_entry_horizon_s
                        and not self._cool(tid, EventType.PEDESTRIAN_APPROACHING_ROAD, frame_idx)):
                    events.append(Event(EventType.PEDESTRIAN_APPROACHING_ROAD, frame_idx,
                                        severity=float(np.clip(1.0 - rng / 40.0, 0.1, 1.0)),
                                        track_id=tid,
                                        payload={"label": t.label, "range_m": rng,
                                                 "lateral_speed": float(closing_lateral),
                                                 "path_entry_s": float(entry_s)}))

        if not events:
            return events

        # GLOBAL RATE LIMIT. Per-track cooldowns bound how often ONE object can
        # speak, but a busy intersection has dozens of tracks and the union still
        # chatters. A real monitor speaks a few times a minute, not hundreds.
        # Keep only the most severe event per frame within the global cooldown.
        if (self._last_any_frame is not None
                and frame_idx - self._last_any_frame < self.global_cooldown_frames):
            return []
        events.sort(key=lambda e: -e.severity)
        self._last_any_frame = frame_idx
        return events[:1]
