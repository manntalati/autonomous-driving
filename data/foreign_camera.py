"""
P13 — simulate a *different camera* so cross-camera robustness can be measured.

WHY THIS EXISTS
---------------
Retraining for "user video" without a way to score it would be optimising for a
domain we cannot observe. This module manufactures a held-out benchmark by
degrading nuScenes val frames the way a foreign camera would, so the question
"does robustness training help on other people's footage?" gets a number instead
of an impression.

It is a PROXY, not the real thing. It reproduces the transferable parts of the
gap — optics, colour pipeline, compression, motion — and cannot reproduce a
different city, different traffic, different mounting, or a rolling shutter. Treat
a gain here as necessary-but-not-sufficient evidence.

The four effects, and why each was chosen:

  FOV        the dominant term. nuScenes is 64.8 deg; a dashcam is 100-140 deg,
             so objects land at roughly half the pixel scale the anchors expect.
             Simulated by cropping to a NARROWER angle then upscaling, which
             mimics the resolution loss the real crop suffers.
  colour     a different ISP: gamma, white balance, saturation. Cheap to simulate
             and the thing photometric augmentation actually fixes.
  compression consumer video is heavily H.264-compressed; nuScenes ships clean
             JPEG. Blocking artifacts change high-frequency statistics.
  motion blur a windscreen mount vibrates and a rolling shutter smears; nuScenes
             is a rigid roof rig.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class ForeignCamera:
    """One simulated camera. Defaults approximate a mid-range windscreen dashcam."""
    name: str = "dashcam"
    hfov_deg: float = 110.0        # wider than nuScenes' 64.8
    gamma: float = 1.25            # brighter/flatter ISP curve
    saturation: float = 1.20
    warmth: float = 1.06           # red gain / blue cut, a warm white balance
    jpeg_quality: int = 45         # aggressive consumer compression
    motion_blur_px: int = 3
    seed: int = 0


PRESETS = {
    # Mild: a decent phone on a windscreen mount.
    "phone": ForeignCamera("phone", hfov_deg=78.0, gamma=1.10, saturation=1.10,
                           warmth=1.03, jpeg_quality=70, motion_blur_px=1),
    # Typical consumer dashcam.
    "dashcam": ForeignCamera("dashcam"),
    # Harsh: wide action camera, heavy compression, visible shake.
    "action_cam": ForeignCamera("action_cam", hfov_deg=140.0, gamma=1.40,
                                saturation=1.35, warmth=1.10, jpeg_quality=30,
                                motion_blur_px=5),
}


def fov_transform(w: int, h: int, hfov_deg: float, native_hfov: float = 64.8):
    """
    The (scale, x_offset, y_offset) that `_apply_fov` applies to image content.

    Exposed separately and computed only from (w, h, hfov) so that GROUND-TRUTH
    BOXES CAN BE MOVED WITH THE PIXELS. Degrading the image while leaving boxes at
    their original coordinates does not measure robustness — it measures
    misalignment, and even a perfect detector scores ~0. Both the image path and
    the label path derive the transform from this one function so they cannot
    drift apart.

    Returns (1.0, 0, 0) when no FOV change applies.
    """
    import math
    if hfov_deg <= native_hfov:
        return 1.0, 0, 0
    ratio = math.tan(math.radians(native_hfov / 2)) / math.tan(math.radians(hfov_deg / 2))
    nw, nh = max(int(w * ratio), 8), max(int(h * ratio), 8)
    return ratio, (w - nw) // 2, (h - nh) // 2


def transform_boxes(boxes, w: int, h: int, hfov_deg: float, native_hfov: float = 64.8):
    """Apply the same scale+offset to [x1, y1, x2, y2] boxes as the image gets."""
    import numpy as _np
    s, x0, y0 = fov_transform(w, h, hfov_deg, native_hfov)
    if s == 1.0:
        return boxes
    b = _np.asarray(boxes, dtype=_np.float64).reshape(-1, 4)
    b[:, [0, 2]] = b[:, [0, 2]] * s + x0
    b[:, [1, 3]] = b[:, [1, 3]] * s + y0
    return b


def _apply_fov(img: np.ndarray, hfov_deg: float, native_hfov: float = 64.8) -> np.ndarray:
    """
    Make `img` look as though it came from a wider-FOV camera.

    A wide camera packs more scene into the same sensor, so each object occupies
    fewer pixels. We cannot add scene that was never captured, so we approximate
    the *scale* effect: shrink the content by the FOV ratio and pad, which lands
    objects at the smaller pixel size a wide lens would produce. The padding is
    the honest signature of a simulation — a real wide lens would show more world
    there.

    Callers MUST push GT boxes through `transform_boxes` with the same arguments.
    """
    if hfov_deg <= native_hfov:
        return img
    h, w = img.shape[:2]
    ratio, x0, y0 = fov_transform(w, h, hfov_deg, native_hfov)
    nw, nh = max(int(w * ratio), 8), max(int(h * ratio), 8)
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros_like(img)
    out[y0:y0 + nh, x0:x0 + nw] = small
    return out


def _apply_photometric(img: np.ndarray, c: ForeignCamera) -> np.ndarray:
    x = img.astype(np.float32) / 255.0
    x = np.power(np.clip(x, 0, 1), 1.0 / max(c.gamma, 1e-3))          # ISP gamma
    if abs(c.warmth - 1.0) > 1e-6:                                     # white balance
        x[..., 0] = np.clip(x[..., 0] * c.warmth, 0, 1)                # R up
        x[..., 2] = np.clip(x[..., 2] / c.warmth, 0, 1)                # B down
    if abs(c.saturation - 1.0) > 1e-6:
        grey = x.mean(axis=2, keepdims=True)
        x = np.clip(grey + (x - grey) * c.saturation, 0, 1)
    return (x * 255).astype(np.uint8)


def _apply_compression(img: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


def _apply_motion_blur(img: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    if k < 2:
        return img
    kernel = np.zeros((k, k), np.float32)
    if rng.random() < 0.5:
        kernel[k // 2, :] = 1.0 / k          # horizontal smear
    else:
        np.fill_diagonal(kernel, 1.0 / k)    # diagonal shake
    return cv2.filter2D(img, -1, kernel)


def simulate(img_rgb: np.ndarray, camera: ForeignCamera,
             rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Apply the full foreign-camera chain to an RGB frame.

    Order matters and follows the physical pipeline: optics first, then the
    sensor/ISP, then motion during exposure, then the codec last — compression
    artifacts sit on top of everything else, as they do in a real file.
    """
    rng = rng or np.random.default_rng(camera.seed)
    out = _apply_fov(img_rgb, camera.hfov_deg)
    out = _apply_photometric(out, camera)
    out = _apply_motion_blur(out, camera.motion_blur_px, rng)
    return _apply_compression(out, camera.jpeg_quality)
