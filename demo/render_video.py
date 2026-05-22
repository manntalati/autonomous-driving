"""
Phase 7 — render an annotated driving video.

nuScenes keyframes are only 2 Hz (choppy). The dense CAM_FRONT *sweep* stream
runs at ~12 Hz — walking the sample_data linked list gives every frame, not
just the annotated keyframes. This module runs the unified perception
pipeline over that dense stream and writes an annotated MP4: the camera view
(detections + segmentation) beside the top-down BEV panel.
"""
from __future__ import annotations
import shutil
import subprocess
from pathlib import Path
import numpy as np
import torch
import cv2
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes

from data.transforms import MEAN, STD, INPUT_H, INPUT_W
from demo.pipeline import PerceptionPipeline
from utils.visualize import draw_boxes, overlay_segmentation, draw_bev

CAM = "CAM_FRONT"
_MEAN = np.array(MEAN, dtype=np.float32)
_STD = np.array(STD, dtype=np.float32)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def scene_sweep_tokens(nusc: NuScenes, scene_name: str) -> list[str]:
    """
    Every CAM_FRONT sample_data token for a scene — keyframes AND sweeps — in
    time order. The sample_data next/prev chain links all frames of one sensor.
    """
    scene = next(s for s in nusc.scene if s["name"] == scene_name)
    sd = nusc.get("sample_data", nusc.get("sample", scene["first_sample_token"])["data"][CAM])
    while sd["prev"] != "":                       # rewind to the scene's first frame
        sd = nusc.get("sample_data", sd["prev"])
    tokens = [sd["token"]]
    while sd["next"] != "":
        sd = nusc.get("sample_data", sd["next"])
        tokens.append(sd["token"])
    return tokens


def _load_frame(nusc: NuScenes, data_root: str, cam_sd_token: str) -> torch.Tensor:
    """Load + resize + ImageNet-normalise a CAM_FRONT image → (3, H, W) tensor."""
    sd = nusc.get("sample_data", cam_sd_token)
    img = Image.open(Path(data_root) / sd["filename"]).convert("RGB")
    img = img.resize((INPUT_W, INPUT_H), Image.BILINEAR)
    arr = (np.asarray(img, dtype=np.float32) / 255.0 - _MEAN) / _STD
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous().float()


def _calibration(nusc: NuScenes, cam_sd_token: str):
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


def _denormalize(image_tensor: torch.Tensor) -> np.ndarray:
    """(3, H, W) normalised tensor → (H, W, 3) uint8 RGB."""
    img = image_tensor.permute(1, 2, 0).cpu().numpy()
    return (np.clip(img * _STD + _MEAN, 0, 1) * 255).astype(np.uint8)


def render_scene_video(cfg: dict, scene_name: str, out_path: str | Path, fps: int = 12, max_frames: int | None = None, pipeline: PerceptionPipeline | None = None) -> Path:
    """
    Run the pipeline over a scene's dense frame stream and write an annotated MP4.
    Args:
      cfg — parsed configs/demo.yaml.
      scene_name — nuScenes scene to render.
      out_path — output .mp4 path.
      fps — playback frame rate.
      max_frames — cap the number of frames (None = whole scene).
      pipeline — a pre-built PerceptionPipeline (built here if None).
    Returns: the output Path.
    """
    device = _device()
    if pipeline is None:
        pipeline = PerceptionPipeline(cfg, device)
    nusc = NuScenes(version="v1.0-mini", dataroot=cfg["data_root"], verbose=False)

    tokens = scene_sweep_tokens(nusc, scene_name)
    if max_frames is not None:
        tokens = tokens[:max_frames]
    frames = [_load_frame(nusc, cfg["data_root"], tk) for tk in tokens]
    seq_len = cfg.get("seq_len", 3)
    xbound, ybound = tuple(pipeline.bev_cfg["xbound"]), tuple(pipeline.bev_cfg["ybound"])

    out_path = Path(out_path)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg not found on PATH — it is needed to encode a browser-playable "
            "(H.264) MP4. Install it (e.g. `brew install ffmpeg`)."
        )

    # Frames are piped to ffmpeg and encoded as H.264 / yuv420p — the codec
    # HTML5 <video> (and Streamlit's st.video) can actually decode. cv2's
    # mp4v writer produces files that play in VLC but stay black in a browser.
    proc = None
    for i, sd_token in enumerate(tokens):
        window = torch.stack([frames[max(0, i - k)] for k in reversed(range(seq_len))])
        intrinsic, cam_to_ego = _calibration(nusc, sd_token)
        out = pipeline.process_frame(window, intrinsic, cam_to_ego)

        cam = _denormalize(frames[i])
        cam = overlay_segmentation(cam, out["seg_mask"].numpy(), alpha=0.45)
        cam = draw_boxes(cam, out["boxes"].tolist(), out["labels"].tolist(), out["scores"].tolist())
        bev = draw_bev(out["bev_boxes"], out["bev_scores"], out["bev_labels"], xbound, ybound)
        bev = cv2.resize(bev, (cam.shape[0], cam.shape[0]))   # square panel, match cam height
        combined = np.ascontiguousarray(np.hstack([cam, bev]), dtype=np.uint8)   # RGB

        if proc is None:
            h, w = combined.shape[:2]
            proc = subprocess.Popen(
                [ffmpeg, "-y", "-loglevel", "error",
                 "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
                 "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path)],
                stdin=subprocess.PIPE,
            )
        proc.stdin.write(combined.tobytes())

    if proc is not None:
        proc.stdin.close()
        proc.wait()
    return out_path


if __name__ == "__main__":
    import sys
    import yaml
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/demo.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    scene = sys.argv[2] if len(sys.argv) > 2 else cfg["scene"]
    out = render_scene_video(cfg, scene, f"demo/{scene}.mp4")
    print(f"wrote {out}")
