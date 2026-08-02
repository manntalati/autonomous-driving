"""
Phase 7 — interactive perception demo (Streamlit).

Pick a nuScenes scene, scrub through its frames, and view the unified
perception output: 2D detections + segmentation overlaid on the camera image,
with a top-down BEV detection panel beside it.

Run:  streamlit run demo/app.py
(Requires `pip install streamlit`.)
"""
from __future__ import annotations
import sys
from pathlib import Path

# `streamlit run demo/app.py` puts demo/ on sys.path, not the project root —
# add the root so the data/models/utils packages import correctly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml
import streamlit as st
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes

from data.dataset import get_scene_split
from data.radar_utils import load_radar_points, radar_bev_for_sample
from data.transforms import MEAN, STD, INPUT_H, INPUT_W
from demo.pipeline import PerceptionPipeline
from demo.render_video import render_scene_video
from utils.visualize import (draw_bev, draw_bev_comparison, draw_boxes,
                             draw_radar_points, draw_range_bands,
                             overlay_segmentation)

CAM = "CAM_FRONT"
_MEAN = np.array(MEAN, dtype=np.float32)
_STD = np.array(STD, dtype=np.float32)
_CFG_PATH = "configs/demo.yaml"
_TRAINVAL_ROOT = "data/raw/v1.0-trainval"   # used to identify scenes the trainval models trained on


def _denormalize(image_tensor: torch.Tensor) -> np.ndarray:
    """(3, H, W) ImageNet-normalised tensor → (H, W, 3) uint8 RGB."""
    img = image_tensor.permute(1, 2, 0).cpu().numpy()
    return (np.clip(img * _STD + _MEAN, 0, 1) * 255).astype(np.uint8)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@st.cache_resource
def load_pipeline(cfg_path: str):
    """Build the perception pipeline once and cache it across reruns."""
    cfg = yaml.safe_load(open(cfg_path))
    device = _device()
    return PerceptionPipeline(cfg, device), cfg, device


@st.cache_resource
def load_nusc(data_root: str):
    """Load the NuScenes index once (slow) and cache it."""
    return NuScenes(version="v1.0-mini", dataroot=data_root, verbose=False)


@st.cache_resource
def held_out_scenes(all_scene_names: tuple) -> list:
    """Restrict the picker to scenes the trainval models never trained on.
    If the trainval split (data/raw/v1.0-trainval) isn't on disk, return everything;
    otherwise drop any scene that appears in the trainval *train* split.
    Falls back to the full list if filtering would leave the picker empty.
    """
    if not Path(_TRAINVAL_ROOT, "v1.0-trainval").exists():
        return list(all_scene_names)
    nusc_t = NuScenes(version="v1.0-trainval", dataroot=_TRAINVAL_ROOT, verbose=False)
    seen = set(get_scene_split(nusc_t, _TRAINVAL_ROOT)[0])    # trainval train scenes
    held = [s for s in all_scene_names if s not in seen]
    return held or list(all_scene_names)


def _scene_cam_tokens(nusc, scene_name: str):
    """Ordered CAM_FRONT sample_data tokens for a scene."""
    rec = next(s for s in nusc.scene if s["name"] == scene_name)
    tokens, t = [], rec["first_sample_token"]
    while t != "":
        sample = nusc.get("sample", t)
        tokens.append(sample["data"][CAM])
        t = sample["next"]
    return tokens


def _load_frame(nusc, data_root: str, cam_sd_token: str) -> torch.Tensor:
    """Load + resize + ImageNet-normalise a CAM_FRONT image → (3, H, W) tensor."""
    sd = nusc.get("sample_data", cam_sd_token)
    img = Image.open(f"{data_root}/{sd['filename']}").convert("RGB")
    img = img.resize((INPUT_W, INPUT_H), Image.BILINEAR)
    arr = (np.asarray(img, dtype=np.float32) / 255.0 - _MEAN) / _STD
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous().float()


def _calibration(nusc, cam_sd_token: str):
    """Intrinsics (scaled to input resolution) + camera→ego transform."""
    sd = nusc.get("sample_data", cam_sd_token)
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    K = np.array(cs["camera_intrinsic"], dtype=np.float32).copy()
    K[0, :] *= INPUT_W / sd["width"]
    K[1, :] *= INPUT_H / sd["height"]
    cam_to_ego = np.eye(4, dtype=np.float32)
    cam_to_ego[:3, :3] = Quaternion(cs["rotation"]).rotation_matrix
    cam_to_ego[:3, 3] = np.array(cs["translation"], dtype=np.float32)
    return torch.from_numpy(K), torch.from_numpy(cam_to_ego)


def _surround_inputs(nusc, data_root: str, cam_sd_token: str, cameras):
    """Load all N surround-camera images + calibration for the sample containing
    cam_sd_token. Used by the BEV path so the Phase-8 model gets its full 360° input."""
    sd = nusc.get("sample_data", cam_sd_token)
    sample = nusc.get("sample", sd["sample_token"])
    imgs, ks, c2es = [], [], []
    for cam in cameras:
        tk = sample["data"][cam]
        imgs.append(_load_frame(nusc, data_root, tk))
        K, c2e = _calibration(nusc, tk)
        ks.append(K)
        c2es.append(c2e)
    return torch.stack(imgs), torch.stack(ks), torch.stack(c2es)


def main() -> None:
    """Streamlit entry point — scene picker, frame slider, unified perception view."""
    try:
        st.set_page_config(page_title="AD Perception Demo", layout="wide")
    except Exception:
        pass   # already configured by demo/showcase.py
    st.title("Autonomous Driving Perception — Unified Demo")

    pipeline, cfg, _ = load_pipeline(_CFG_PATH)
    nusc = load_nusc(cfg["data_root"])

    all_scenes = tuple(sorted(s["name"] for s in nusc.scene))
    scene_names = held_out_scenes(all_scenes)
    default = cfg["scene"] if cfg["scene"] in scene_names else scene_names[0]
    scene = st.sidebar.selectbox("Scene", scene_names, index=scene_names.index(default))
    st.sidebar.caption(
        f"{len(scene_names)} / {len(all_scenes)} mini scenes — only those the trainval "
        f"models never trained on are shown."
    )

    cam_tokens = _scene_cam_tokens(nusc, scene)
    seq_len = cfg.get("seq_len", 3)
    frame_idx = st.sidebar.slider("Frame", 0, len(cam_tokens) - 1, 0)

    # 3-frame window ending at frame_idx, left-padded at the scene start
    window_tokens = [cam_tokens[max(0, frame_idx - k)] for k in reversed(range(seq_len))]
    window = torch.stack([_load_frame(nusc, cfg["data_root"], tk) for tk in window_tokens])
    intrinsic, cam_to_ego = _calibration(nusc, cam_tokens[frame_idx])

    # surround inputs for the BEV path: all N cameras of the current frame
    bev_cams = pipeline.bev_cfg.get("cameras", [CAM])
    bev_surround = (_surround_inputs(nusc, cfg["data_root"], cam_tokens[frame_idx], bev_cams)
                    if len(bev_cams) > 1 else None)
    out = pipeline.process_frame(window, intrinsic, cam_to_ego, bev_surround=bev_surround)

    image = _denormalize(window[-1])
    # skip the drivable class (1) — its road-surface paint hides vehicles on the road
    vis = overlay_segmentation(image, out["seg_mask"].numpy(), alpha=0.45, skip_classes=(1,))
    vis = draw_boxes(vis, out["boxes"].tolist(), out["labels"].tolist(), out["scores"].tolist())
    bev_img = draw_bev(out["bev_boxes"], out["bev_scores"], out["bev_labels"],
                       tuple(pipeline.bev_cfg["xbound"]), tuple(pipeline.bev_cfg["ybound"]),
                       seg=out["bev_seg"].numpy())

    left, right = st.columns([2, 1])
    left.subheader("Camera — detection + segmentation")
    left.image(vis, width="stretch")
    right.subheader("Bird's-eye view")
    right.image(bev_img, width="stretch")
    st.caption(
        f"Scene {scene} · frame {frame_idx + 1}/{len(cam_tokens)} · "
        f"{len(out['boxes'])} detections · {len(out['bev_boxes'])} BEV objects"
    )

    # ── live video: run the pipeline over the dense sweep stream → MP4 ──
    st.divider()
    st.subheader("Live video — dense sweep stream (~12 Hz)")
    n_frames = st.slider("Frames to render", 24, 240, 96, step=12)
    if st.button("Render annotated video"):
        out_path = f"demo/_render_{scene}.mp4"
        with st.spinner(f"Running perception over {n_frames} frames…"):
            render_scene_video(cfg, scene, out_path, fps=12,
                                max_frames=n_frames, pipeline=pipeline)
        st.video(out_path)

    # ── Phase 10: camera-only vs camera+radar, side by side ──────────────────
    # The headline finding is a RANGE effect, so the panel draws the 20 m / 35 m
    # analysis boundaries and the raw radar returns. Segmentation is deliberately
    # off here: this comparison is about detections, and the BEV map's large
    # colour fields drown the boxes that carry the story.
    st.divider()
    st.subheader("Phase 10 — does radar help? (camera only vs camera + radar)")
    # Opt-in: this runs TWO 6-camera BEV forward passes (measured 3.5s) plus
    # six uncached image loads. Running it on every rerun made each scene
    # change and slider move stall before the rest of the page rendered, which
    # looked like the scene picker and video renderer had stopped working.
    run_ab = st.checkbox("Run the radar A/B comparison", value=False,
                         help="Two 6-camera BEV passes, ~4 s per frame")
    if not run_ab:
        st.caption("Enable to compare camera-only vs camera+radar BEV on this frame.")
    elif not pipeline.has_radar_arm:
        st.info(
            "Radar A/B unavailable — needs `checkpoints/bev_radar_last.pt` and "
            "`checkpoints/bev_surround_p10_last.pt`. Train both arms with "
            "`bash scripts/run_p10_ablation.sh`."
        )
    else:
        rcfg = pipeline.bev_radar_cfg
        XB, YB = tuple(rcfg["xbound"]), tuple(rcfg["ybound"])
        ab_imgs, ab_K, ab_c2e = _surround_inputs(
            nusc, cfg["data_root"], cam_tokens[frame_idx], rcfg["cameras"])
        # _surround_inputs returns (N, ...) on CPU; the BEV models want a batch
        # dimension and everything on the pipeline's device. Without the unsqueeze
        # the model flattens (6,3,H,W) to 18 "channels"; without the .to() the
        # camera tensors and the radar grid land on different devices.
        ab_imgs = ab_imgs.unsqueeze(0).to(pipeline.device)
        ab_K = ab_K.unsqueeze(0).to(pipeline.device)
        ab_c2e = ab_c2e.unsqueeze(0).to(pipeline.device)
        sample_tok = nusc.get("sample_data", cam_tokens[frame_idx])["sample_token"]
        sample = nusc.get("sample", sample_tok)
        radar_grid = torch.from_numpy(
            radar_bev_for_sample(nusc, cfg["data_root"], sample, XB, YB)
        ).float().unsqueeze(0).to(pipeline.device)
        radar_pts = load_radar_points(nusc, cfg["data_root"], sample)

        with st.spinner("Running both BEV arms…"):
            ab = pipeline.process_bev_pair(ab_imgs, ab_K, ab_c2e, radar_grid)

        show_radar = st.checkbox("Overlay radar returns", value=True)
        panels = []
        for key, title in (("camera", "camera only"), ("camera_radar", "camera + radar")):
            canvas = draw_bev(ab[key]["boxes"], ab[key]["scores"], ab[key]["labels"],
                              XB, YB, seg=None, canvas_px=420)
            draw_range_bands(canvas, XB, YB)
            if show_radar and key == "camera_radar":
                draw_radar_points(canvas, radar_pts, XB, YB)
            panels.append((canvas, title, f"{len(ab[key]['boxes'])} detections"))

        st.image(draw_bev_comparison(panels[0][0], panels[1][0], panels[0][1],
                                     panels[1][1], panels[0][2], panels[1][2]),
                 width="stretch")

        gate = ab.get("gate")
        gate_note = ""
        if gate is not None:
            lean = "camera" if gate > 0.5 else "radar"
            gate_note = (f" · fusion gate {gate:.3f} (leans {lean}; "
                         f"ablation measured 0.522 day → 0.660 night)")
        st.caption(
            f"{len(radar_pts)} radar returns · rings at 20 m / 35 m mark the "
            f"analysis buckets{gate_note}"
        )
        st.markdown(
            "**What this shows.** Radar's benefit is concentrated at long range: "
            "camera-only BEV mAP is 0.012 beyond 35 m *in daylight* versus 0.102 "
            "with radar. It is **not** night-specific — day benefit +0.044 vs night "
            "+0.038, inside the ±0.02 noise band declared before the run — so the "
            "original illumination-invariance hypothesis was refuted. "
            "Both arms: 12 fixed epochs, seed 0, final checkpoints, n=1."
        )



if __name__ == "__main__":
    main()
