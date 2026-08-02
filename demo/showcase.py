"""
Unified showcase — the single entry point for the whole project.

    streamlit run demo/showcase.py

Sections:
    Overview        the Phase 9-13 investigation and what it found
    Live demo       detection + segmentation + BEV on held-out nuScenes scenes
    Your own video  upload a clip and run the stack on it (Phase 13)

The agent demo stays a separate app (`streamlit run demo/agent_app.py`) because it
configures its page at import time and needs an ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import streamlit as st

FIG = Path("docs/figures")


def _fig(stem: str, caption: str = "") -> None:
    """Show a result figure if it has been generated (tools/make_figures.py)."""
    p = FIG / f"{stem}_light.png"
    if p.exists():
        st.image(str(p), width="stretch")
        if caption:
            st.caption(caption)
    else:
        st.info(f"`{p}` not found — run `python -m tools.make_figures` to generate it.")


def _load(path: str):
    try:
        return json.load(open(path))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def overview() -> None:
    st.title("A perception stack that knows when it's wrong")
    st.markdown(
        "Phases 0–8 built a full autonomous-driving perception stack from scratch: "
        "detection, segmentation, bird's-eye-view, temporal fusion, and an agent API. "
        "**Phases 9–13 are an investigation into where it breaks.**"
    )

    st.divider()
    st.header("The finding that started it")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(
            """
The training data is **3,376 keyframes of clear daylight**. Evaluated on held-out
daytime frames the detector reaches **mAP 0.285**. On night footage it has never
seen, it collapses to **0.095** — a 67% drop, cyclists worst at −87%.

Nothing warns you. The detector reports the same confidence scores it always did.
That silent failure, not raw accuracy, is the open problem in AV deployment.

A control rules out the obvious objection: the night scenes come from a different
dataset split, so is the drop just data provenance? No — *daytime* scenes from that
same split score **0.304**, statistically indistinguishable from the 0.285 baseline.
The collapse tracks lighting, not source.
            """
        )
    with c2:
        _fig("day_night")

    st.divider()
    st.header("Three hypotheses, each with a pre-registered failure condition")
    st.caption(
        "Each was written down — including what result would refute it — before the "
        "run. Two were refuted. That is the point of writing them down first."
    )

    h1, h2, h3 = st.columns(3)
    with h1:
        st.markdown("#### 1 · Radar closes the night gap")
        st.error("**Refuted**")
        st.markdown(
            "Radar measures range and velocity directly, so it should be "
            "illumination-invariant. Test: night benefit ≫ day benefit.\n\n"
            "**Result:** +0.038 night vs **+0.044 day** — a 0.006 difference, inside "
            "the ±0.02 band declared in advance. The learned fusion gate agreed "
            "independently, moving *toward* the camera at night (0.52 → 0.66)."
        )
    with h2:
        st.markdown("#### 2 · Epistemic uncertainty predicts failure")
        st.error("**Refuted**")
        st.markdown(
            "MC-dropout variance should flag detections the model is unsure of. "
            "Test: does it beat the raw confidence score?\n\n"
            "**Result:** +0.004 AUROC at night against a **±0.006** confidence "
            "interval. Noise. Cause traced to `score_var` ≈ 2e-05 — dropout sits "
            "only in the head, so the sampled posterior is far too narrow."
        )
    with h3:
        st.markdown("#### 3 · Robustness training helps user video")
        st.warning("**Confirmed, then overturned**")
        st.markdown(
            "Augmentation should close the cross-camera gap. It did: **+14%**.\n\n"
            "**But** re-scoring through the path the demo actually runs — which "
            "normalises field of view first — a free inference-time crop delivers "
            "**+31%**, and the two together give +32%. The two-hour retrain adds "
            "~1% on top, and costs 8% native accuracy."
        )

    st.divider()
    st.header("What survived")
    s1, s2 = st.columns(2)
    with s1:
        st.markdown("#### Radar closes the *range* gap")
        _fig("range_gap")
        st.markdown(
            "Camera-only BEV mAP is **0.012 beyond 35 m in daylight** — depth "
            "inference has effectively failed. Radar holds **0.102**, an 8.2× gain, "
            "and 12.4× at night. Not the hypothesis, but a cleaner mechanism: radar "
            "measures range instead of inferring it. Only the range-stratified "
            "evaluation made this visible; the overall average hides it entirely."
        )
    with s2:
        st.markdown("#### The trust layer is calibrated")
        _fig("calibration")
        st.markdown(
            "The introspection head reaches **ECE 0.012** on night footage against "
            "the raw score's **0.063** — 5× better, and unlike accuracy it holds "
            "across the domain boundary. Its night calibration is no worse than its "
            "day calibration."
        )

    st.divider()
    st.header("And what it still can't do")
    st.markdown(
        """
- **Night BEV detection remains near-unusable at 0.044 mAP.** Radar improved it 7.7×,
  but from a near-zero baseline. The problem is not solved.
- **Abstention buys little at night**, because 97% of night detections are false
  positives — no ranking of those rescues much.
- **Every result is n=1.** Two runs of an identical config produced epoch-1 mIoU of
  0.124 and 0.185, so seed variance is real. Establishing effect sizes needs 3+ seeds.
- **The cross-camera benchmark is a proxy.** It reproduces optics, ISP, codec and
  motion — not a different city, traffic mix, or mounting.
        """
    )

    st.divider()
    st.header("The recurring lesson")
    st.markdown(
        """
Three separate times, a result turned out to be an artifact of *how it was measured*:

1. **Phase 10** early-stopped on segmentation mIoU while reporting detection mAP —
   and mIoU was noise-dominated, so "best epoch" was a coin flip.
2. **Phase 13's benchmark** degraded images without moving the ground-truth boxes.
   A *perfect* detector would have scored zero.
3. **Phase 13's retrain** was benchmarked on the model in isolation, while the demo
   preprocesses first — which reversed the conclusion.

Each was caught by checking the measurement before believing it. The fix was the
same every time: **evaluate the pipeline you deploy, not the component you changed.**
        """
    )

    with st.expander("Raw result files"):
        for name, path in (
            ("Day/night audit", "logs/day_night_audit.json"),
            ("Radar 2×2 ablation", "logs/radar_ablation.json"),
            ("Introspection (no epistemic)", "logs/introspection_nomc.json"),
            ("Introspection (MC-dropout)", "logs/introspection_mc.json"),
            ("Cross-camera, raw", "logs/foreign_camera_eval.json"),
            ("Cross-camera, FOV-normalised", "logs/foreign_camera_eval_norm.json"),
        ):
            st.write(f"`{path}` — {'✅' if _load(path) else '— not generated'} {name}")


def live_demo() -> None:
    from demo.app import main as app_main
    app_main()


def own_video() -> None:
    from demo.byo_app import main as byo_main
    byo_main()


def main() -> None:
    st.set_page_config(page_title="AD Perception — Showcase", layout="wide",
                       initial_sidebar_state="expanded")
    nav = st.navigation([
        st.Page(overview, title="Overview", icon="🔍", default=True),
        st.Page(live_demo, title="Live demo (nuScenes)", icon="🎥"),
        st.Page(own_video, title="Your own video", icon="📹"),
    ])
    nav.run()


main()
