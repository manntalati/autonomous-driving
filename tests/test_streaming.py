"""Tests for P12 — tracker, event salience, advisory memory, abstention."""
import asyncio
import math

import numpy as np
import pytest

from agent.monitor import EventDetector, EventType, SceneTracker
from agent.streaming_agent import AdvisoryMemory, Advisory, StreamingAgent


def yaw_q(a):
    return [math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)]


def ego(x=0.0, y=0.0, yaw=0.0):
    return {"translation": [x, y, 0.0], "rotation": yaw_q(yaw)}


def drive(y_lat, gx0, dgx, n=10, label=0, ego_speed=1.0):
    """Replay a scenario; returns (events, tracker). ego moves +ego_speed m per 0.1 s."""
    tr, ed, out = SceneTracker(), EventDetector(), []
    for i in range(n):
        e = ego(i * ego_speed, 0, 0)
        gx = gx0 + dgx * i
        det = np.array([[gx - i * ego_speed, y_lat, 4.0, 2.0, 0.0]])
        tracks = tr.update(det, [label], [0.9], int(i * 1e5), i, ego_pose=e)
        out += ed.step(tracks, i)
    return out, tr


class TestSceneTracker:
    def test_single_object_one_track(self):
        _, tr = drive(0.0, 30.0, 0.0)
        assert len(tr.tracks) == 1

    def test_parked_car_has_zero_absolute_velocity(self):
        _, tr = drive(8.0, 40.0, 0.0)
        t = list(tr.tracks.values())[0]
        assert np.allclose(t.velocity_ego, [0.0, 0.0], atol=1e-6)

    def test_parked_car_has_nonzero_relative_velocity(self):
        """Relative velocity is what collides: the ego is closing at its own speed."""
        _, tr = drive(8.0, 40.0, 0.0)
        t = list(tr.tracks.values())[0]
        assert t.relative_velocity_ego[0] == pytest.approx(-10.0, abs=0.5)

    def test_class_change_does_not_associate(self):
        tr = SceneTracker()
        tr.update(np.array([[10.0, 0, 4, 2, 0]]), [0], [0.9], 0, 0, ego_pose=ego())
        tr.update(np.array([[10.0, 0, 4, 2, 0]]), [1], [0.9], 100000, 1, ego_pose=ego())
        assert len(tr.tracks) == 2

    def test_track_expires_after_max_missed(self):
        tr = SceneTracker(max_missed=2)
        tr.update(np.array([[10.0, 0, 4, 2, 0]]), [0], [0.9], 0, 0, ego_pose=ego())
        for i in range(1, 5):
            tr.update(np.zeros((0, 5)), [], [], int(i * 1e5), i, ego_pose=ego())
        assert len(tr.tracks) == 0

    def test_empty_detections_safe(self):
        tr = SceneTracker()
        assert tr.update(np.zeros((0, 5)), [], [], 0, 0, ego_pose=ego()) == {}


class TestEventSalience:
    def test_parked_car_beside_road_is_silent(self):
        """Line-of-sight TTC shrinks for anything you drive past; gate on the corridor."""
        events, _ = drive(8.0, 40.0, 0.0)
        assert [e for e in events if e.type == EventType.TTC_BELOW_THRESHOLD] == []

    def test_stationary_obstacle_in_path_warns(self):
        events, _ = drive(0.0, 30.0, 0.0)
        assert any(e.type == EventType.TTC_BELOW_THRESHOLD for e in events)

    def test_receding_object_never_warns(self):
        events, _ = drive(0.0, 20.0, 2.0)
        assert events == []

    def test_lost_object_far_away_is_not_reported(self):
        """An unrestricted rule made 73 of 136 events cars fading out past 40 m.
        Lateral offset must be outside the ego corridor, or the object counts as
        in-path and stays relevant no matter how distant."""
        tr, ed = SceneTracker(), EventDetector()
        for i in range(4):
            tr.update(np.array([[45.0, 12.0, 4, 2, 0]]), [0], [0.9], int(i * 1e5), i, ego_pose=ego())
            ed.step(tr.tracks, i)
        tr.update(np.zeros((0, 5)), [], [], int(4 * 1e5), 4, ego_pose=ego())
        assert [e for e in ed.step(tr.tracks, 4) if e.type == EventType.OBJECT_LOST] == []

    def test_lost_object_in_path_is_reported_at_any_range(self):
        """Losing track of something in your lane matters regardless of distance."""
        tr, ed = SceneTracker(), EventDetector()
        for i in range(4):
            tr.update(np.array([[45.0, 0.5, 4, 2, 0]]), [0], [0.9], int(i * 1e5), i, ego_pose=ego())
            ed.step(tr.tracks, i)
        tr.update(np.zeros((0, 5)), [], [], int(4 * 1e5), 4, ego_pose=ego())
        assert any(e.type == EventType.OBJECT_LOST for e in ed.step(tr.tracks, 4))

    def test_lost_object_nearby_is_reported(self):
        tr, ed = SceneTracker(), EventDetector()
        for i in range(4):
            tr.update(np.array([[10.0, 0.5, 4, 2, 0]]), [0], [0.9], int(i * 1e5), i, ego_pose=ego())
            ed.step(tr.tracks, i)
        tr.update(np.zeros((0, 5)), [], [], int(4 * 1e5), 4, ego_pose=ego())
        assert any(e.type == EventType.OBJECT_LOST for e in ed.step(tr.tracks, 4))

    def test_trust_events_fire_on_transition_only(self):
        ed = EventDetector()
        seq = [{"in_odd": True, "trust": 0.9}, {"in_odd": False, "trust": 0.2},
               {"in_odd": False, "trust": 0.2}, {"in_odd": True, "trust": 0.8}]
        fired = [e.type for i, t in enumerate(seq) for e in ed.step({}, i * 20, trust=t)]
        assert fired == [EventType.TRUST_DEGRADED, EventType.TRUST_RECOVERED]

    def test_global_cooldown_caps_rate(self):
        """Per-track cooldowns bound one object; a busy scene still chatters."""
        ed = EventDetector(global_cooldown_frames=5)
        tr = SceneTracker()
        n = 0
        for i in range(10):
            dets = np.array([[10.0 + j, 0.0, 4, 2, 0] for j in range(8)])
            tr.update(dets, [0] * 8, [0.9] * 8, int(i * 1e5), i, ego_pose=ego(i * 1.0))
            n += len(ed.step(tr.tracks, i))
        assert n <= 3

    def test_at_most_one_event_per_frame(self):
        ed = EventDetector()
        tr = SceneTracker()
        dets = np.array([[10.0 + j, 0.0, 4, 2, 0] for j in range(6)])
        tr.update(dets, [0] * 6, [0.9] * 6, 0, 0, ego_pose=ego())
        assert len(ed.step(tr.tracks, 0)) <= 1


class TestAdvisoryMemory:
    def _ev(self, frame, sev=0.5, tid=1):
        from agent.monitor import Event
        return Event(EventType.OBJECT_ENTERED_PATH, frame, sev, track_id=tid)

    def test_first_event_not_suppressed(self):
        assert not AdvisoryMemory().should_suppress(self._ev(0))

    def test_repeat_within_ttl_suppressed(self):
        m = AdvisoryMemory(ttl_frames=30)
        e = self._ev(0)
        m.record(Advisory(0, "x", e.type.value, 1.0, False), e)
        assert m.should_suppress(self._ev(5))

    def test_escalation_breaks_through(self):
        m = AdvisoryMemory(ttl_frames=30, escalation=0.25)
        e = self._ev(0, sev=0.4)
        m.record(Advisory(0, "x", e.type.value, 1.0, False), e)
        assert not m.should_suppress(self._ev(5, sev=0.9))

    def test_expires_after_ttl(self):
        m = AdvisoryMemory(ttl_frames=10)
        e = self._ev(0)
        m.record(Advisory(0, "x", e.type.value, 1.0, False), e)
        assert not m.should_suppress(self._ev(50))

    def test_different_track_not_suppressed(self):
        m = AdvisoryMemory()
        e = self._ev(0, tid=1)
        m.record(Advisory(0, "x", e.type.value, 1.0, False), e)
        assert not m.should_suppress(self._ev(2, tid=2))

    def test_recent_context_when_empty(self):
        assert "nothing said" in AdvisoryMemory().recent_context()


class TestAbstention:
    def _agent(self):
        return StreamingAgent(mcp_client=None, min_severity=0.0)

    def test_out_of_odd_abstains_without_llm(self):
        """Abstention is a Python branch, not a prompt instruction."""
        agent = self._agent()
        trust = {"trust": 0.1, "in_odd": False, "reason": "low light"}
        out = asyncio.run(agent.step(np.array([[10.0, 0.0, 4, 2, 0]]), [0], [0.9],
                                     0, 0, ego_pose=ego(), trust=trust))
        # force an event on the next frame by moving the object into the path
        out += asyncio.run(agent.step(np.array([[8.0, 0.0, 4, 2, 0]]), [0], [0.9],
                                      100000, 1, ego_pose=ego(1.0), trust=trust))
        abstained = [a for a in out if a.abstained]
        assert abstained, "expected a degraded-perception advisory"
        assert "degraded" in abstained[0].text.lower()
        assert agent.stats["llm_calls"] == 0

    def test_in_odd_does_not_abstain(self):
        agent = self._agent()
        trust = {"trust": 0.95, "in_odd": True}
        asyncio.run(agent.step(np.array([[30.0, 0.0, 4, 2, 0]]), [0], [0.9],
                               0, 0, ego_pose=ego(), trust=trust))
        out = asyncio.run(agent.step(np.array([[20.0, 0.0, 4, 2, 0]]), [0], [0.9],
                                     100000, 1, ego_pose=ego(1.0), trust=trust))
        assert all(not a.abstained for a in out)

    def test_severity_gate_blocks_llm(self):
        agent = StreamingAgent(mcp_client=None, min_severity=1.1)   # nothing qualifies
        asyncio.run(agent.step(np.array([[10.0, 0.0, 4, 2, 0]]), [0], [0.9],
                               0, 0, ego_pose=ego()))
        out = asyncio.run(agent.step(np.array([[5.0, 0.0, 4, 2, 0]]), [0], [0.9],
                                     100000, 1, ego_pose=ego(1.0)))
        assert out == []

    def test_rate_summary_shape(self):
        agent = self._agent()
        asyncio.run(agent.step(np.zeros((0, 5)), [], [], 0, 0, ego_pose=ego()))
        r = agent.rate_summary(60.0)
        assert {"frames", "events_per_min", "llm_calls", "abstentions"} <= set(r)
