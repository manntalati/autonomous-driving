"""
P13 — Bring-your-own-video inference.

Run the perception stack on footage the user supplies, rather than on nuScenes.

WHAT TRANSFERS AND WHAT DOES NOT
--------------------------------
    detection      works — resize + normalise is all it needs
    segmentation   works — same
    BEV            needs camera intrinsics AND camera->ego extrinsics, neither of
                   which a phone or dashcam file carries. Both are ESTIMATED here
                   from user-supplied FOV / mount geometry, and every BEV output
                   from this path must be labelled as resting on assumed geometry.
    radar          impossible — there is no sensor.

THE SINGLE BIGGEST FIX IS GEOMETRIC, NOT LEARNED
------------------------------------------------
The detector's anchor scales (32/64/128 px at 448x800) are calibrated for
nuScenes CAM_FRONT, measured at **64.8 deg horizontal FOV** (fx = 1261 px at
1600 px wide). A typical dashcam is 100-140 deg — it sees roughly twice as wide,
so every object lands at about half the pixel scale the anchors expect, and the
detector misses systematically.

`fov_normalize` fixes this with a centre crop that keeps exactly the angular
extent the model was trained on, then resizes. No retraining required, and it
recovers far more than any amount of colour augmentation would.

    crop_fraction = tan(target_hfov / 2) / tan(source_hfov / 2)

    120 deg dashcam -> keep 36.6% of the width
     73 deg phone   -> keep 85.7%

EXPECT THE TRUST LAYER TO FIRE
------------------------------
Phase 9 measured a 67% mAP collapse moving from daytime to night *within one
dataset*. Arbitrary user footage — different camera, optics, colour pipeline,
mounting — is further out of distribution than that. The honest expectation is
that the stack degrades and the Phase 11 trust layer reports OUTSIDE ODD. That
is the system working, not failing: this mode is the project's thesis applied to
the user's own data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np
import torch

from data.transforms import INPUT_H, INPUT_W, MEAN, STD

# nuScenes CAM_FRONT, measured from calibrated_sensor: fx 1261 px at 1600 px wide.
NUSCENES_HFOV_DEG = 64.8
NUSCENES_CAM_HEIGHT_M = 1.51
NUSCENES_CAM_FORWARD_M = 1.70

# Rough horizontal FOV presets, for users who do not know their camera's optics.
FOV_PRESETS = {
    "Phone, main lens (~70°)": 70.0,
    "Phone, ultra-wide (~106°)": 106.0,
    "Dashcam, typical (~120°)": 120.0,
    "Dashcam, narrow (~100°)": 100.0,
    "GoPro wide (~120°)": 120.0,
    "GoPro superview (~150°)": 150.0,
    "Match nuScenes (64.8°) — no crop": NUSCENES_HFOV_DEG,
}


@dataclass
class CameraAssumption:
    """User-supplied (or guessed) geometry. Everything downstream is only as good as this."""
    hfov_deg: float = 120.0
    height_m: float = NUSCENES_CAM_HEIGHT_M
    forward_m: float = NUSCENES_CAM_FORWARD_M
    pitch_deg: float = 0.0        # positive = nose down

    def describe(self) -> str:
        return (f"assumed {self.hfov_deg:.0f}° HFOV, mounted {self.height_m:.2f} m high, "
                f"{self.pitch_deg:+.1f}° pitch")


def crop_fraction(source_hfov_deg: float, target_hfov_deg: float = NUSCENES_HFOV_DEG) -> float:
    """
    Width fraction to keep so the crop subtends `target_hfov_deg`.

    Returns 1.0 when the source is already at or narrower than the target — we
    never up-crop, because that would invent field of view that was not captured.
    """
    if source_hfov_deg <= target_hfov_deg:
        return 1.0
    return math.tan(math.radians(target_hfov_deg / 2)) / math.tan(math.radians(source_hfov_deg / 2))


def fov_normalize(frame: np.ndarray, source_hfov_deg: float,
                  target_hfov_deg: float = NUSCENES_HFOV_DEG,
                  out_hw: Tuple[int, int] = (INPUT_H, INPUT_W)) -> np.ndarray:
    """
    Centre-crop `frame` to the training FOV, then resize to the model's input size.

    Args: frame — (H, W, 3) BGR or RGB uint8; source_hfov_deg — the camera's true
      horizontal FOV; out_hw — model input (default 448x800).
    Returns: (out_h, out_w, 3) uint8.

    The crop is centred horizontally and vertically. A dashcam is usually mounted
    near the optical centre, so a centre crop keeps the road ahead; a camera aimed
    high or low would want a vertical offset, which `pitch_deg` covers on the BEV
    side but cannot fix in the image.
    """
    h, w = frame.shape[:2]
    f = crop_fraction(source_hfov_deg, target_hfov_deg)
    if f < 1.0:
        cw = max(int(round(w * f)), 16)
        # keep the output aspect ratio so the resize does not stretch geometry
        ch = max(int(round(cw * out_hw[0] / out_hw[1])), 16)
        ch = min(ch, h)
        x0 = (w - cw) // 2
        y0 = (h - ch) // 2
        frame = frame[y0:y0 + ch, x0:x0 + cw]
    return cv2.resize(frame, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_AREA)


def estimate_intrinsics(hfov_deg: float, out_hw: Tuple[int, int] = (INPUT_H, INPUT_W)) -> np.ndarray:
    """
    K for the POST-normalisation image.

    After `fov_normalize` the image subtends the nuScenes FOV regardless of the
    source, so K is the nuScenes-equivalent intrinsic at the model's input size —
    which is exactly what the BEV transform was trained against.
    """
    h, w = out_hw
    fx = (w / 2) / math.tan(math.radians(NUSCENES_HFOV_DEG / 2))
    return np.array([[fx, 0.0, w / 2.0],
                     [0.0, fx, h / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float32)


def assumed_extrinsics(a: CameraAssumption) -> np.ndarray:
    """
    4x4 camera->ego transform from assumed mount geometry.

    Built to nuScenes' convention, verified against the real calibration:
        camera z (forward) -> ego +x
        camera x (right)   -> ego -y
        camera y (down)    -> ego -z
    with the measured CAM_FRONT sitting at [1.70, 0.02, 1.51] m.

    Pitch rotates about the camera's x axis (nose down positive), which is the
    one mount error that most distorts BEV placement: a few degrees of pitch
    moves the ground-plane intersection by metres at range.
    """
    R = np.array([[0.0, 0.0, 1.0],
                  [-1.0, 0.0, 0.0],
                  [0.0, -1.0, 0.0]], dtype=np.float64)
    p = math.radians(a.pitch_deg)
    # rotation about the camera x axis, applied in camera coordinates
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, math.cos(p), -math.sin(p)],
                   [0.0, math.sin(p), math.cos(p)]], dtype=np.float64)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = (R @ Rx).astype(np.float32)
    T[:3, 3] = np.array([a.forward_m, 0.0, a.height_m], dtype=np.float32)
    return T


def normalize_for_model(frame_rgb: np.ndarray) -> torch.Tensor:
    """(H, W, 3) uint8 RGB -> (3, H, W) float tensor, ImageNet-normalised."""
    x = frame_rgb.astype(np.float32) / 255.0
    x = (x - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    return torch.from_numpy(x).permute(2, 0, 1).contiguous().float()


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    n_frames: int
    is_portrait: bool

    def summary(self) -> str:
        orient = "PORTRAIT" if self.is_portrait else "landscape"
        return (f"{self.width}x{self.height} {orient} · {self.fps:.1f} fps · "
                f"{self.n_frames} frames · {self.n_frames / max(self.fps, 1e-6):.1f}s")


def probe_video(path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return VideoInfo(str(path), w, h, fps, n, is_portrait=h > w)


def iter_frames(path: str | Path, assumption: CameraAssumption,
                stride: int = 1, max_frames: Optional[int] = None
                ) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
    """
    Yield (frame_index, raw_rgb, normalised_rgb) for the video.

    `stride` subsamples (a 30 fps clip at stride 15 gives 2 Hz, matching the
    keyframe rate the models were trained and evaluated at).

    Yields the RAW frame alongside the normalised one so the UI can show what the
    model actually sees after FOV cropping — which is often a surprise, since a
    120° dashcam loses ~63% of its width.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    idx = emitted = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                yield idx, rgb, fov_normalize(rgb, assumption.hfov_deg)
                emitted += 1
                if max_frames and emitted >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()


def backbone_of(model: torch.nn.Module) -> Optional[torch.nn.Module]:
    """
    The image backbone inside a detector/segmenter/BEV model, or None.

    Adaptation targets the backbone rather than the whole model for two reasons.
    It is where the domain shift actually enters — the first convolutions see raw
    pixel statistics — and it is the one submodule with a plain (B, 3, H, W)
    contract. The temporal detector takes a (B, T, 3, H, W) window and the BEV
    model takes (B, N, 3, H, W) plus calibration, so feeding either a single frame
    silently misinterprets the tensor's axes.

    Searches one wrapper deep, because TemporalDetector holds its FPNDetector in
    `self.detector` and only that inner model exposes `.backbone`. Returning the
    wrapper instead would feed a single frame to a model expecting a window, which
    fails with a confusing channel-count error rather than an obvious one.
    """
    direct = getattr(model, "backbone", None)
    if isinstance(direct, torch.nn.Module):
        return direct
    for attr in ("detector", "segmenter", "bev", "encoder", "model"):
        inner = getattr(model, attr, None)
        if isinstance(inner, torch.nn.Module):
            found = getattr(inner, "backbone", None)
            if isinstance(found, torch.nn.Module):
                return found
    enc = getattr(model, "encoder", None)
    return enc if isinstance(enc, torch.nn.Module) else None


@torch.no_grad()
def adapt_batchnorm(model: torch.nn.Module, frames: list[torch.Tensor],
                    device: torch.device, momentum: float = 0.1) -> int:
    """
    Test-time BatchNorm adaptation (AdaBN) on the user's own footage.

    Args: model — a detector/segmenter (its backbone is located automatically) or
      any module accepting (B, 3, H, W); frames — normalised (3, H, W) tensors
      from the user's video; momentum — BN running-stat update rate.
    Returns: number of BatchNorm layers adapted (0 if no backbone was found).

    WHY THIS HELPS AND WHY IT IS SAFE
    ---------------------------------
    The backbone's BN running statistics were estimated on nuScenes. A different
    camera and colour pipeline shifts the activation distribution, so those stored
    means and variances are simply wrong for this input — the layer normalises by
    the wrong constants before any weight is applied.

    Re-estimating the statistics on the user's frames needs NO LABELS: it only
    puts BN modules in train mode so their running estimates update on forward
    passes, while every weight stays frozen and no gradient is computed. This is
    AdaBN (Li et al., 2016), a standard first move for domain shift.

    It mutates the model in place, so callers that need the original statistics
    must keep their own copy — `PerceptionPipeline` reloads from checkpoint.
    """
    target = backbone_of(model) or model
    bns = [m for m in target.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bns or not frames:
        return 0
    model.eval()
    saved = []
    for m in bns:
        saved.append(m.momentum)
        m.momentum = momentum
        m.train()          # ONLY BN — weights stay frozen, no grad is taken
    try:
        for f in frames:
            target(f.unsqueeze(0).to(device))
    finally:
        # Restore eval mode even if a forward fails, or the model is left
        # stochastic for every subsequent frame the user scrubs to.
        for m, mom in zip(bns, saved):
            m.momentum = mom
            m.eval()
    return len(bns)


# ── P13 per-camera model selection ──────────────────────────────────────────
# Measured on the simulated-foreign-camera benchmark WITH FOV normalisation
# applied (i.e. the deployed path), mean over 513 held-out frames:
#
#   camera        base+norm   robust+norm
#   phone  78 deg    0.2794       0.2562     <- baseline wins
#   dashcam 110 deg  0.2296       0.2387     <- robust wins, marginally
#   action 140 deg   0.0234       0.0420     <- robust wins clearly
#
# The robust checkpoint's advantage GROWS with field of view: a wide camera needs
# a tighter crop, which upscales heavily, and the robustness run trained on
# exactly that degradation (Downscale / compression / blur). Below ~80 deg the
# crop is mild, the input is near-native, and the robust model's 8% native-accuracy
# cost dominates — so the plain checkpoint is the better choice there.
#
# The crossover is placed at 90 deg: between the phone (78) and dashcam (110)
# measurements, where the ordering flips. With only three sampled FOVs this is an
# interpolation, not a measured boundary — worth re-measuring at 90-100 deg before
# treating it as precise.
DETECTOR_FOV_CROSSOVER_DEG = 90.0

DETECTOR_CHOICES = {
    "baseline": ("configs/detector.yaml", "checkpoints/detector_best.pt"),
    "robust": ("configs/detector_robust.yaml", "checkpoints/detector_robust_last.pt"),
}


def select_detector(hfov_deg: float) -> tuple[str, str, str, str]:
    """
    Pick the detector checkpoint suited to this camera's field of view.

    Returns (name, config_path, checkpoint_path, rationale).

    Falls back to the baseline if the robust checkpoint has not been trained, so
    the demo still runs on a fresh clone.
    """
    from pathlib import Path as _P

    wide = hfov_deg >= DETECTOR_FOV_CROSSOVER_DEG
    name = "robust" if wide else "baseline"
    cfg, ckpt = DETECTOR_CHOICES[name]
    if not _P(ckpt).exists():
        name, (cfg, ckpt) = "baseline", DETECTOR_CHOICES["baseline"]
        return name, cfg, ckpt, "robust checkpoint not found — using the baseline"
    if wide:
        why = (f"{hfov_deg:.0f}° needs a tight crop, so the frame is upscaled a lot; "
               f"the robustness-trained model handles that degradation better "
               f"(measured 0.042 vs 0.023 at 140°)")
    else:
        why = (f"{hfov_deg:.0f}° crops only mildly, leaving near-native input where "
               f"the plain checkpoint is stronger (measured 0.279 vs 0.256 at 78°)")
    return name, cfg, ckpt, why
