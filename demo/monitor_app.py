"""
P12 — Streaming autonomy monitor demo.

    streamlit run demo/monitor_app.py

Plays a scene frame by frame through the two-tier loop and shows what the monitor
would have said during the drive — including the frames where it declined to say
anything because perception had left its operating domain.

The fast tier runs on every frame with no LLM. The slow tier is only reached when
a deterministic event fires AND trust is inside the ODD, which is what keeps the
API cost proportional to what actually happened rather than to the frame rate.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import torch
import yaml
from nuscenes.nuscenes import NuScenes

from agent.monitor import EventDetector, EventType, SceneTracker
from agent.streaming_agent import StreamingAgent
from data.dataset import NIGHT_SCENES
from demo.app import _calibration, _scene_cam_tokens, load_nusc, load_pipeline
from demo.stream import FrameStream
from models.uncertainty.introspection import TrustScorer

_CFG = "configs/demo.yaml"
_TRUST_CKPT = "checkpoints/introspection_nomc.pt"

_EVENT_ICON = {
    EventType.OBJECT_ENTERED_PATH.value: "🚗",
    EventType.TTC_BELOW_THRESHOLD.value: "⚠️",
    EventType.PEDESTRIAN_APPROACHING_ROAD.value: "🚶",
    EventType.OBJECT_LOST.value: "❓",
    EventType.TRUST_DEGRADED.value: "🔴",
    EventType.TRUST_RECOVERED.value: "🟢",
}


@st.cache_resource
def _trust_scorer(path: str):
    return TrustScorer.load(path) if Path(path).exists() else None


def main() -> None:
    try:
        st.set_page_config(page_title="Streaming autonomy monitor", layout="wide")
    except Exception:
        pass   # already configured by demo/showcase.py

    st.title("Streaming autonomy monitor")
    st.caption(
        "A continuously running monitor that watches the drive and speaks only when "
        "something matters — and abstains when perception is outside its operating domain."
    )

    pipeline, cfg, device = load_pipeline(_CFG)
    nusc = load_nusc(cfg["data_root"])
    scorer = _trust_scorer(_TRUST_CKPT)

    scenes = sorted(s["name"] for s in nusc.scene)
    default = "scene-1094" if "scene-1094" in scenes else scenes[0]
    scene = st.sidebar.selectbox("Scene", scenes, index=scenes.index(default))
    is_night = scene in NIGHT_SCENES
    st.sidebar.caption("🌙 night scene — expect the trust layer to fire"
                       if is_night else "☀️ daytime scene")

    n_frames = st.sidebar.slider("Frames to monitor", 4, 40, 16)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Fast tier thresholds")
    ttc = st.sidebar.slider("TTC warn threshold (s)", 1.0, 8.0, 3.5, 0.5)
    cooldown = st.sidebar.slider("Global cooldown (frames)", 1, 15, 4)
    st.sidebar.caption(
        "The cooldown is what keeps the monitor quiet. Without it the fast tier "
        "produced 418 events/min on a real scene; with it, 18.5."
    )

    if scorer is None:
        st.warning(
            f"No trust layer at `{_TRUST_CKPT}` — the monitor will run without "
            "abstention. Train it with "
            "`python -m models.uncertainty.train_introspection configs/detector_dropout.yaml`."
        )

    if not st.button("Run the monitor", type="primary"):
        _explain()
        return

    stream = FrameStream(nusc, cfg["data_root"], scene, mode="keyframes", load_images=True)
    K, c2e = _calibration(nusc, _scene_cam_tokens(nusc, scene)[0])

    agent = StreamingAgent(
        pipeline=pipeline,
        tracker=SceneTracker(),
        detector=EventDetector(ttc_enter_s=ttc, global_cooldown_frames=cooldown),
        trust_scorer=scorer,
        mcp_client=None,          # fast tier only; see the note below
        min_severity=0.0,
    )

    prog = st.progress(0.0, text="Running perception frame by frame…")
    advisories = asyncio.run(agent.run(stream, K, c2e, seq_len=cfg.get("seq_len", 3),
                                       max_frames=n_frames))
    prog.empty()

    meta = list(FrameStream(nusc, cfg["data_root"], scene, mode="keyframes",
                            load_images=False))[:n_frames]
    dur = (meta[-1].timestamp_us - meta[0].timestamp_us) / 1e6 if len(meta) > 1 else 1.0
    rates = agent.rate_summary(dur)

    c = st.columns(5)
    c[0].metric("Frames", agent.stats["frames"])
    c[1].metric("Events", agent.stats["events"])
    c[2].metric("Advisories", len(advisories))
    c[3].metric("Abstentions", agent.stats["abstentions"])
    c[4].metric("Events / min", f"{rates['events_per_min']:.1f}")

    quiet = 1.0 - len({a.frame_idx for a in advisories}) / max(agent.stats["frames"], 1)
    st.caption(
        f"**{quiet*100:.0f}% of frames produced nothing.** That is the design goal — "
        f"a monitor that narrates every frame is unusable. LLM calls this run: "
        f"{rates['llm_calls']} (the slow tier is skipped here; see below)."
    )

    st.divider()
    st.subheader("What the monitor said, frame by frame")
    if not advisories:
        st.info("Nothing salient happened in this window — no events crossed threshold.")
    for a in advisories:
        icon = "🔇" if a.abstained else _EVENT_ICON.get(a.event_type, "•")
        with st.container(border=True):
            cols = st.columns([1, 6, 2])
            cols[0].markdown(f"### {icon}")
            cols[0].caption(f"frame {a.frame_idx}")
            cols[1].markdown(f"**{a.text}**")
            cols[1].caption(f"`{a.event_type}`")
            colour = "🔴" if a.trust < 0.5 else ("🟡" if a.trust < 0.75 else "🟢")
            cols[2].metric("Trust", f"{colour} {a.trust:.2f}")
            if a.abstained:
                cols[2].caption("outside ODD")

    _explain()


def _explain() -> None:
    st.divider()
    st.subheader("How this works")
    st.markdown(
        """
**Two tiers, because an LLM in the frame loop is the wrong architecture.**
At 12 Hz that is ~1,900 API calls per scene, seconds of latency per frame, and an
agent that says "car ahead" 1,900 times. Production stacks run a cheap
deterministic safety monitor at high rate and reserve expensive reasoning for when
something warrants it.

- **Fast tier** — every frame, pure numpy, no LLM. Tracks objects across frames
  with ego-motion compensation, computes time-to-collision from *relative*
  velocity, and raises typed events on **change and risk**, never on presence.
  A car sitting 30 m ahead in a steady state produces nothing, ever.
- **Slow tier** — reached only when an event fires. Turns the event into one
  sentence via the MCP perception tools.

**Abstention is a Python branch, not a prompt instruction.** When trust falls below
the ODD threshold the language model is never given the chance to make a confident
object-level claim — a templated degraded-perception advisory is emitted instead,
citing which signal deviated. A prompt rule is a strong suggestion; a branch is a
guarantee, and for a safety-relevant behaviour the guarantee is what you want.

**Why the LLM is off here.** This page runs the fast tier and the abstention logic,
which is where the interesting engineering is, and prints a deterministic
description of each event instead of calling the API. Enabling the slow tier needs
an `ANTHROPIC_API_KEY` with credits; the fast tier's behaviour — what fires, what
stays quiet, when it abstains — is identical either way.

**Known limitation.** The trust layer scores the 2-D detections it was trained on,
so its risk weighting is class-based rather than range-weighted. Range-weighted
trust needs a BEV-trained introspection head, which does not exist yet.
        """
    )


if __name__ == "__main__":
    main()
