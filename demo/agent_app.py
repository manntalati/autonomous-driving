"""Unified demo: trained perception models + agent over MCP.

Layout: trained det/seg/BEV overlays render on the current frame (left); the
agent, given the same scene+frame in its question, hits the MCP tool API and
streams its reasoning + final answer (right). Both views look at the same
underlying nuScenes frame, so the agent's claims are visually verifiable.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import torch
from nuscenes.nuscenes import NuScenes

from agent.config import load_env
from agent.loop import AgentResult, run_agent
from agent.mcp_client import MCPClient
from agent.prompts import SYSTEM_PROMPT
from data.dataset import MINI_VAL_SCENES
from demo.agent_data import load_surround_images, scene_length
from demo.app import (
    CAM,
    _calibration,
    _denormalize,
    _load_frame,
    _scene_cam_tokens,
    _surround_inputs,
    load_nusc,
    load_pipeline,
)
from demo.render_video import render_scene_video
from utils.visualize import draw_bev, draw_boxes, overlay_segmentation

_CFG_PATH  = "configs/demo.yaml"
_PRICE_IN  = 15.0 / 1_000_000
_PRICE_OUT = 75.0 / 1_000_000

PRESETS = [
    ("Count cars",         "How many cars are in {scene} frame {frame}?"),
    ("Lane safety (left)", "Is it safe to change into the left lane in {scene} frame {frame}?"),
    ("Should I stop?",     "Should I stop in {scene} frame {frame}?"),
    ("Following distance", "How far is the car in front of me in {scene} frame {frame}?"),
    ("Describe scene",     "Describe the scene at {scene} frame {frame}."),
]

st.set_page_config(page_title="AD Perception + Agent", layout="wide")


# ─── caches ──────────────────────────────────────────────────────────────────

@st.cache_data
def cached_scene_length(scene_name: str, data_root: str) -> int:
    return scene_length(load_nusc(data_root), scene_name)

@st.cache_data
def cached_surround(scene_name: str, frame_idx: int, data_root: str):
    return load_surround_images(load_nusc(data_root), scene_name, frame_idx)

@st.cache_data(show_spinner="Running trained perception models…")
def run_perception(scene: str, frame_idx: int):
    """Run det+seg+BEV on the current frame; return (overlay_img, bev_img, n_dets, n_bev).
    Cached by (scene, frame_idx) so scrubbing the slider is fast on repeat visits.
    """
    pipeline, cfg, _ = load_pipeline(_CFG_PATH)
    nusc = load_nusc(cfg["data_root"])
    cam_tokens = _scene_cam_tokens(nusc, scene)
    seq_len = cfg.get("seq_len", 3)

    window_tokens = [cam_tokens[max(0, frame_idx - k)] for k in reversed(range(seq_len))]
    window = torch.stack([_load_frame(nusc, cfg["data_root"], tk) for tk in window_tokens])
    intrinsic, cam_to_ego = _calibration(nusc, cam_tokens[frame_idx])

    bev_cams = pipeline.bev_cfg.get("cameras", [CAM])
    bev_surround = (_surround_inputs(nusc, cfg["data_root"], cam_tokens[frame_idx], bev_cams)
                    if len(bev_cams) > 1 else None)
    out = pipeline.process_frame(window, intrinsic, cam_to_ego, bev_surround=bev_surround)

    image = _denormalize(window[-1])
    vis   = overlay_segmentation(image, out["seg_mask"].numpy(), alpha=0.45, skip_classes=(1,))
    vis   = draw_boxes(vis, out["boxes"].tolist(), out["labels"].tolist(), out["scores"].tolist())
    bev   = draw_bev(out["bev_boxes"], out["bev_scores"], out["bev_labels"],
                     tuple(pipeline.bev_cfg["xbound"]), tuple(pipeline.bev_cfg["ybound"]),
                     seg=out["bev_seg"].numpy())
    return vis, bev, len(out["boxes"]), len(out["bev_boxes"])


# ─── ui helpers ──────────────────────────────────────────────────────────────

def _init_session_state() -> None:
    load_env()
    st.session_state.setdefault("trace_events", [])
    st.session_state.setdefault("agent_result", None)
    st.session_state.setdefault("last_latency", 0.0)

def _sidebar(data_root: str) -> tuple[str, int, str, bool]:
    st.sidebar.header("Scene")
    scenes = sorted(MINI_VAL_SCENES)
    scene  = st.sidebar.selectbox("Scene", scenes, index=0)
    n_frames = cached_scene_length(scene, data_root)
    frame_idx = st.sidebar.slider("Frame", 0, max(0, n_frames - 1), 0)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Quick questions**")
    triggered = None
    for label, template in PRESETS:
        if st.sidebar.button(label, key=f"preset_{label}", use_container_width=True):
            triggered = template.format(scene=scene, frame=frame_idx)

    st.sidebar.markdown("---")
    question   = st.sidebar.text_area("Or ask your own", value="", key="custom_q", height=100)
    ask_custom = st.sidebar.button("Ask", type="primary", use_container_width=True)

    if triggered:
        return scene, frame_idx, triggered, True
    return scene, frame_idx, question, ask_custom

def _render_surround_grid(images: dict) -> None:
    order = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
             "CAM_BACK_LEFT",  "CAM_BACK",  "CAM_BACK_RIGHT"]
    row1 = st.columns(3)
    row2 = st.columns(3)
    for col, cam in zip(row1, order[:3]):
        col.image(images[cam], caption=cam, use_container_width=True)
    for col, cam in zip(row2, order[3:]):
        col.image(images[cam], caption=cam, use_container_width=True)

def _compact_args(args: dict) -> str:
    if not args:
        return ""
    parts = []
    for k, v in args.items():
        s = str(v)
        if len(s) > 30:
            s = s[:27] + "..."
        parts.append(f"{k}={s}")
    return ", ".join(parts)

def _format_trace_md(events: list[dict]) -> str:
    blocks = []
    for e in events:
        t = e["type"]
        if t == "text":
            blocks.append(f"*thinking* — {e['text']}")
        elif t == "tool_call":
            blocks.append(f"**call** `{e['name']}({_compact_args(e['input'])})`")
        elif t == "tool_result":
            out = e["output"]
            if len(out) > 280:
                out = out[:280] + "…"
            blocks.append(f"**result** `{out}`")
        elif t == "final":
            blocks.append(f"**answer** — {e['answer']}")
    return "\n\n---\n\n".join(blocks) if blocks else ""

def _render_metrics_strip(result: AgentResult | None, latency: float) -> None:
    cols = st.columns(4)
    if result is None:
        for c, name in zip(cols, ["Latency", "Turns", "Tokens", "Cost"]):
            c.metric(name, "—")
        return
    cost = result.input_tokens * _PRICE_IN + result.output_tokens * _PRICE_OUT
    cols[0].metric("Latency", f"{latency:.1f}s")
    cols[1].metric("Turns",   result.turns)
    cols[2].metric("Tokens",  result.input_tokens + result.output_tokens)
    cols[3].metric("Cost",    f"${cost:.4f}")


# ─── agent driver ────────────────────────────────────────────────────────────

async def _run_agent_once(question: str, on_event) -> AgentResult:
    client = MCPClient()
    await client.connect()
    try:
        return await run_agent(question, client, system=SYSTEM_PROMPT, on_event=on_event)
    finally:
        await client.close()


# ─── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    _init_session_state()
    pipeline, cfg, _ = load_pipeline(_CFG_PATH)
    data_root = cfg["data_root"]

    st.title("AD Perception + Agent")
    st.caption(
        "Left: trained detector + segmenter + BEV running in-process on this frame. "
        "Right: the agent independently calls the same models via MCP tools and explains the scene."
    )

    scene, frame_idx, question, ask = _sidebar(data_root)

    # Surround context — collapsed by default; opens with one click during a demo.
    with st.expander("Surround cameras (all 6)", expanded=False):
        surround = cached_surround(scene, frame_idx, data_root)
        _render_surround_grid(surround)

    col_ml, col_trace = st.columns([3, 2])

    with col_ml:
        st.markdown("### Trained-model perception")
        vis, bev, n_dets, n_bev = run_perception(scene, frame_idx)
        sub_cam, sub_bev = st.columns([2, 1])
        sub_cam.image(vis, caption=f"CAM_FRONT — {n_dets} detections + segmentation", use_container_width=True)
        sub_bev.image(bev, caption=f"BEV — {n_bev} objects",                          use_container_width=True)

    with col_trace:
        st.markdown("### Agent trace (via MCP)")
        trace_placeholder = st.empty()
        prev_events = st.session_state.get("trace_events", [])
        if prev_events:
            trace_placeholder.markdown(_format_trace_md(prev_events))
        else:
            trace_placeholder.info("Pick a preset or ask a question — tool calls stream here.")

    st.markdown("---")
    _render_metrics_strip(
        st.session_state.get("agent_result"),
        st.session_state.get("last_latency", 0.0),
    )

    st.markdown("---")
    st.subheader("Annotated video — dense sweep stream (~12 Hz)")
    st.caption("Runs the same trained perception models over a contiguous window of frames and writes an MP4.")
    n_frames = st.slider("Frames to render", 24, 240, 96, step=12)
    if st.button("Render annotated video"):
        out_path = f"demo/_render_{scene}.mp4"
        with st.spinner(f"Running perception over {n_frames} frames…"):
            render_scene_video(cfg, scene, out_path, fps=12,
                               max_frames=n_frames, pipeline=pipeline)
        st.video(out_path)

    if ask and question.strip():
        events: list[dict] = []

        def on_event(ev: dict) -> None:
            events.append(ev)
            trace_placeholder.markdown(_format_trace_md(events))

        t0 = time.perf_counter()
        try:
            result = asyncio.run(_run_agent_once(question, on_event))
            st.session_state["agent_result"] = result
        except Exception as e:
            st.session_state["agent_result"] = None
            events.append({"type": "text", "text": f"ERROR: {e}"})
            trace_placeholder.markdown(_format_trace_md(events))
        st.session_state["last_latency"] = time.perf_counter() - t0
        st.session_state["trace_events"]  = events
        st.rerun()


if __name__ == "__main__":
    main()
