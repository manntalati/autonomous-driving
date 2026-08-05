"""
P13 — Bring-your-own-video demo.

    streamlit run demo/byo_app.py

Upload a clip, run detection + segmentation + BEV on it, and watch the Phase 11
trust layer react to footage the models have never seen. The expected outcome is
degradation plus an OUTSIDE-ODD flag — that is the thesis working on real input,
not the demo failing.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import streamlit as st
import torch
import yaml

from demo.byo_video import (FOV_PRESETS, NUSCENES_HFOV_DEG, CameraAssumption,
                            adapt_batchnorm, assumed_extrinsics, crop_fraction,
                            estimate_intrinsics, iter_frames, normalize_for_model,
                            probe_video, select_detector)
from demo.pipeline import PerceptionPipeline
from utils.visualize import (draw_bev, draw_boxes, draw_range_bands,
                             draw_trust_banner, overlay_segmentation)

_CFG = "configs/demo.yaml"


@st.cache_resource
def _load(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    dev = (torch.device("mps") if torch.backends.mps.is_available()
           else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    return cfg, PerceptionPipeline(cfg, dev), dev


@st.cache_resource
def _load_detector(ckpt: str, cfg_path: str, _dev_str: str):
    """
    Load the SINGLE-FRAME detector chosen for this camera's FOV.

    The shared PerceptionPipeline runs the Phase 6 temporal detector, but every
    cross-camera number in Phase 13 was measured on the single-frame FPNDetector —
    the temporal model was never benchmarked outside nuScenes. Running the
    temporal model here while quoting single-frame measurements would attribute
    evidence to a model it does not describe, so the BYO path uses the models the
    evidence actually covers.
    """
    from models.detection.train_detector import build_detector

    dcfg = yaml.safe_load(open(cfg_path))
    dev = torch.device(_dev_str)
    model = build_detector(dcfg).to(dev)
    state = torch.load(ckpt, map_location=dev, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    return model, dcfg


def main() -> None:
    try:
        st.set_page_config(page_title="Perception on your own video", layout="wide")
    except Exception:
        pass   # already configured by demo/showcase.py
    st.title("Run the perception stack on your own video")
    st.caption(
        "Detection and segmentation transfer directly. BEV needs camera geometry "
        "your file does not carry, so it runs on **assumed** calibration. Radar is "
        "unavailable — there is no sensor."
    )

    cfg, pipeline, device = _load(_CFG)

    up = st.file_uploader("Video file", type=["mp4", "mov", "avi", "mkv", "m4v"])
    if up is None:
        st.info(
            "Upload a driving clip to begin. A forward-facing dashcam or "
            "phone-on-the-windscreen view works best — the models were trained on "
            "a roof-mounted forward camera 1.5 m above the road."
        )
        _expectations()
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(up.name).suffix) as f:
        f.write(up.read())
        path = f.name
    info = probe_video(path)
    st.success(f"**{up.name}** — {info.summary()}")
    if info.is_portrait:
        st.warning(
            "Portrait video. The models expect a 16:9 landscape road scene, so a "
            "centre crop will discard most of the frame. Landscape footage will "
            "work considerably better."
        )

    # ── camera geometry ────────────────────────────────────────────────────
    st.sidebar.header("Camera geometry")
    st.sidebar.caption(
        "Everything the BEV panel shows rests on these numbers. They are "
        "assumptions, not measurements."
    )
    preset = st.sidebar.selectbox("Field of view", list(FOV_PRESETS), index=2)
    hfov = st.sidebar.slider("Horizontal FOV (°)", 40.0, 170.0,
                             float(FOV_PRESETS[preset]), 1.0)
    height = st.sidebar.slider("Mount height (m)", 0.5, 3.0, 1.51, 0.01)
    pitch = st.sidebar.slider("Pitch (° nose-down)", -15.0, 15.0, 0.0, 0.5)
    assumption = CameraAssumption(hfov_deg=hfov, height_m=height, pitch_deg=pitch)

    frac = crop_fraction(hfov)
    if frac < 1.0:
        msg = (f"FOV normalisation keeps **{frac*100:.0f}%** of the frame width, so the "
               f"crop subtends {NUSCENES_HFOV_DEG}° — the FOV the detector's anchor "
               f"scales were calibrated for.")
        (st.sidebar.error if frac < 0.25 else st.sidebar.info)(
            msg + ("  At this FOV the crop is extreme and very little of your "
                   "footage survives." if frac < 0.25 else ""))
    else:
        st.sidebar.info("Source FOV is at or below the training FOV — no crop applied.")

    # ── run controls ───────────────────────────────────────────────────────
    st.sidebar.header("Run")
    stride = st.sidebar.slider("Frame stride", 1, 60,
                               max(1, int(round(info.fps / 2)) or 1))
    st.sidebar.caption(f"≈ {info.fps / max(stride,1):.1f} Hz — the models were "
                       f"trained and evaluated at 2 Hz keyframes.")
    n_frames = st.sidebar.slider("Frames to process", 1, 120, 12)
    use_adabn = st.sidebar.checkbox("Adapt BatchNorm to this video", value=True)
    st.sidebar.caption(
        "Re-estimates BatchNorm statistics on your footage. Needs no labels and "
        "changes no weights — it just stops the network normalising by constants "
        "measured on a different camera."
    )

    if not st.button("Run perception", type="primary"):
        _expectations()
        return

    frames = list(iter_frames(path, assumption, stride=stride, max_frames=n_frames))
    if not frames:
        st.error("No frames decoded — the file may be corrupt or an unsupported codec.")
        return

    # Per-camera model selection (P13). Which checkpoint wins depends on FOV, and
    # the crossover was measured, not guessed — see select_detector.
    det_name, det_cfg_path, det_ckpt, det_why = select_detector(hfov)
    detector, det_cfg = _load_detector(det_ckpt, det_cfg_path, str(device))
    st.info(f"**Detector: `{det_name}`** — {det_why}")

    if use_adabn:
        with st.spinner("Adapting BatchNorm statistics…"):
            tensors = [normalize_for_model(n) for _, _, n in frames[:min(16, len(frames))]]
            n_bn = adapt_batchnorm(detector, tensors, device)
            adapt_batchnorm(pipeline.segmenter, tensors, device)
        st.caption(f"Adapted {n_bn} BatchNorm layers in the detector on "
                   f"{len(tensors)} of your frames.")

    # process_frame documents (3, 3) and (4, 4). Passing (1, 3, 3) happened to work
    # because it reshapes by element count downstream, but it violated the contract
    # and would break the moment that reshape changed.
    K = torch.from_numpy(estimate_intrinsics(hfov)).to(device)
    T = torch.from_numpy(assumed_extrinsics(assumption)).to(device)
    xb = tuple(pipeline.bev_cfg["xbound"]); yb = tuple(pipeline.bev_cfg["ybound"])
    seq_len = cfg.get("seq_len", 3)

    idx = st.slider("Frame", 0, len(frames) - 1, 0)
    _, raw, norm = frames[idx]
    frame_t = normalize_for_model(norm).unsqueeze(0).to(device)

    with st.spinner("Running perception…"):
        # Detection: the FOV-selected single-frame model, run directly.
        with torch.no_grad():
            boxes_l, scores_l, labels_l = detector(frame_t)
        boxes, scores, labels = boxes_l[0].cpu(), scores_l[0].cpu(), labels_l[0].cpu()
        keep = scores >= cfg.get("det_score_threshold", 0.3)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        # Segmentation and BEV still come from the shared pipeline. The temporal
        # detector it also holds is unused here — see _load_detector.
        window = torch.stack([normalize_for_model(frames[max(0, idx - k)][2])
                              for k in reversed(range(seq_len))]).to(device)
        out = pipeline.process_frame(window, K, T)
    out["boxes"], out["scores"], out["labels"] = boxes, scores, labels

    vis = overlay_segmentation(norm.copy() if norm.dtype == np.uint8 else norm,
                               out["seg_mask"].numpy(), alpha=0.45, skip_classes=(1,))
    vis = draw_boxes(vis, out["boxes"].tolist(), out["labels"].tolist(),
                     out["scores"].tolist())
    bev = draw_bev(out["bev_boxes"], out["bev_scores"], out["bev_labels"], xb, yb,
                   seg=None, canvas_px=420)
    draw_range_bands(bev, xb, yb)

    left, right = st.columns([2, 1])
    left.subheader("What the model sees (after FOV normalisation)")
    left.image(vis, width="stretch")
    right.subheader("BEV — assumed geometry")
    right.image(bev, width="stretch")
    right.caption(assumption.describe())

    n_det = len(out["boxes"])
    st.image(draw_trust_banner(
        900, min(1.0, n_det / 8.0), False,
        "user footage is outside the nuScenes training domain - treat all output as indicative"
    ), width="stretch")
    st.caption(f"{n_det} detections · {len(out['bev_boxes'])} BEV objects · "
               f"frame {idx + 1}/{len(frames)}")

    with st.expander("Original frame, before FOV normalisation"):
        st.image(raw, width="stretch")
        st.caption(f"{raw.shape[1]}x{raw.shape[0]} → cropped to {frac*100:.0f}% width "
                   f"→ resized to 800x448")

    _expectations()


def _expectations() -> None:
    st.divider()
    st.subheader("What to expect, honestly")
    st.markdown(
        """
Phase 9 measured a **67% mAP collapse** moving from daytime to night *inside one
dataset*. Your footage is a different camera, different optics, different colour
pipeline and a different mounting — further out of distribution than that. So:

- **Detection and segmentation will be noticeably worse** than the demo on nuScenes.
- **BEV geometry is only as good as the numbers in the sidebar.** A few degrees of
  pitch error moves the ground-plane intersection by metres at range.
- **Radar cannot run at all** — the Phase 10 range-gap result needs a sensor your
  video does not have, and camera-only BEV is where mAP falls to 0.012 beyond 35 m.

Two things are working in your favour, both free:

- **FOV normalisation** crops to the 64.8° the anchors were calibrated for. Without
  it, a 120° dashcam presents every object at roughly half the expected scale.
- **BatchNorm adaptation** re-estimates normalisation statistics on your frames,
  with no labels and no weight changes.

If the stack looks unreliable on your video, that is the honest reading — and it is
what the trust layer is for.
        """
    )


if __name__ == "__main__":
    main()
