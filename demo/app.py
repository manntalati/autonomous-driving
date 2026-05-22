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

from data.transforms import MEAN, STD, INPUT_H, INPUT_W
from demo.pipeline import PerceptionPipeline
from demo.render_video import render_scene_video
from utils.visualize import draw_boxes, overlay_segmentation, draw_bev

CAM = "CAM_FRONT"
_MEAN = np.array(MEAN, dtype=np.float32)
_STD = np.array(STD, dtype=np.float32)
_CFG_PATH = "configs/demo.yaml"


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


def main() -> None:
    """Streamlit entry point — scene picker, frame slider, unified perception view."""
    st.set_page_config(page_title="AD Perception Demo", layout="wide")
    st.title("Autonomous Driving Perception — Unified Demo")

    pipeline, cfg, _ = load_pipeline(_CFG_PATH)
    nusc = load_nusc(cfg["data_root"])

    scene_names = sorted(s["name"] for s in nusc.scene)
    default = cfg["scene"] if cfg["scene"] in scene_names else scene_names[0]
    scene = st.sidebar.selectbox("Scene", scene_names, index=scene_names.index(default))

    cam_tokens = _scene_cam_tokens(nusc, scene)
    seq_len = cfg.get("seq_len", 3)
    frame_idx = st.sidebar.slider("Frame", 0, len(cam_tokens) - 1, 0)

    # 3-frame window ending at frame_idx, left-padded at the scene start
    window_tokens = [cam_tokens[max(0, frame_idx - k)] for k in reversed(range(seq_len))]
    window = torch.stack([_load_frame(nusc, cfg["data_root"], tk) for tk in window_tokens])
    intrinsic, cam_to_ego = _calibration(nusc, cam_tokens[frame_idx])

    out = pipeline.process_frame(window, intrinsic, cam_to_ego)

    image = _denormalize(window[-1])
    vis = overlay_segmentation(image, out["seg_mask"].numpy(), alpha=0.45)
    vis = draw_boxes(vis, out["boxes"].tolist(), out["labels"].tolist(), out["scores"].tolist())
    bev_img = draw_bev(out["bev_boxes"], out["bev_scores"], out["bev_labels"],
                       tuple(pipeline.bev_cfg["xbound"]), tuple(pipeline.bev_cfg["ybound"]),
                       seg=out["bev_seg"].numpy())

    left, right = st.columns([2, 1])
    left.subheader("Camera — detection + segmentation")
    left.image(vis, use_container_width=True)
    right.subheader("Bird's-eye view")
    right.image(bev_img, use_container_width=True)
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


if __name__ == "__main__":
    main()
