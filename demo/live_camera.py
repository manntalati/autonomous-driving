"""
A-1 — Live camera, v0: phone camera in, perception overlays out, on the laptop.

    python -m demo.live_camera --list              # find the phone's device index
    python -m demo.live_camera --camera 1 --hfov 70 --mount-height 1.3

Point an iPhone at the road via Continuity Camera (macOS 13+; the phone shows up
as an ordinary webcam) and see detection, segmentation and BEV live. This is the
Phase 13 bring-your-own-video path (`demo/byo_app.py`) fed from a camera instead of
a file, so it inherits that path's measured choices:

    detection      the single-frame detector `select_detector(hfov)` picks: the
                   configuration P13 actually benchmarked on foreign cameras
    segmentation   `PerceptionPipeline`, on the FOV-normalised frame
    BEV            `PerceptionPipeline`, on ASSUMED geometry: intrinsics from the
                   FOV guess, extrinsics from the mount height and pitch you type in

WHAT v0 DOES NOT DO YET (each is its own Track A ticket)
--------------------------------------------------------
    real calibration (A-4, A-5) — `--hfov` is a guess. Continuity Camera may
        also switch lenses or crop (Center Stage); turn Center Stage off in the
        macOS Video Effects menu, or the FOV changes under you.
    scheduling (A-3) — every model runs on every frame, so the frame rate is
        whatever the slowest model allows. `process_frame` also still runs the
        temporal detector, whose boxes are discarded in favour of the P13 one.
    AdaBN warm-up (A-6), trust score (A-8), recording (A-9).

The status strip therefore always reads OUTSIDE ODD and shows no trust number.
A placeholder number would look like a measurement, and this path has none yet.

THE TEMPORAL WINDOW IS SAMPLED BY TIME, NOT BY FRAME COUNT
----------------------------------------------------------
The models were trained on 2 Hz keyframes, so a 3-frame window spans 1.0 s. A
camera delivers 30 fps; the last three frames span 67 ms and are nearly
identical. `TemporalWindow` keeps a timestamped ring buffer and picks the frames
closest to 0.5 s apart, whatever rate frames actually arrive at (that part of A-2
lands here).

SAFETY: a passenger runs the laptop, the phone is mounted legally, and nothing
on screen is ever used to make a driving decision.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import deque
from typing import Deque, Generic, List, Optional, Tuple, TypeVar

import cv2
import numpy as np
import torch

from demo.byo_video import (CameraAssumption, assumed_extrinsics, crop_fraction,
                            estimate_intrinsics, fov_normalize, normalize_for_model,
                            select_detector)
from utils.visualize import (_ascii, draw_bev, draw_boxes, draw_range_bands,
                             overlay_segmentation)

WINDOW_TITLE = "perception - live (q to quit)"
TRAINING_SPACING_S = 0.5            # nuScenes keyframes are 2 Hz

T = TypeVar("T")


class TemporalWindow(Generic[T]):
    """
    Timestamped ring buffer that serves a `seq_len` window spaced `spacing_s` apart.

    `sample()` returns items oldest -> newest. Item k is the newest frame taken at
    or before `t_now - (seq_len - 1 - k) * spacing_s`. Until the buffer spans the
    full window (the first second of a session), the oldest frame stands in for
    any slot nothing is old enough to fill. That repeats a frame rather than
    inventing one.

    Memory is bounded by the window span, not the session length: a frame is
    dropped once a newer frame is already old enough to serve the oldest slot.
    """

    def __init__(self, seq_len: int = 3, spacing_s: float = TRAINING_SPACING_S) -> None:
        if seq_len < 1 or spacing_s <= 0:
            raise ValueError("seq_len must be >= 1 and spacing_s > 0")
        self.seq_len = seq_len
        self.spacing_s = spacing_s
        self._buf: Deque[Tuple[float, T]] = deque()

    @property
    def span_s(self) -> float:
        return (self.seq_len - 1) * self.spacing_s

    def __len__(self) -> int:
        return len(self._buf)

    def push(self, t_s: float, item: T) -> None:
        if self._buf and t_s < self._buf[-1][0]:
            raise ValueError("timestamps must be non-decreasing")
        self._buf.append((t_s, item))
        cutoff = t_s - self.span_s
        while len(self._buf) >= 2 and self._buf[1][0] <= cutoff:
            self._buf.popleft()

    @property
    def warm(self) -> bool:
        """True once the buffer spans the full window, so no slot is a stand-in."""
        return bool(self._buf) and self._buf[0][0] <= self._buf[-1][0] - self.span_s

    def sample(self) -> List[T]:
        if not self._buf:
            raise ValueError("TemporalWindow is empty")
        t_now = self._buf[-1][0]
        out: List[T] = []
        for k in reversed(range(self.seq_len)):
            target = t_now - k * self.spacing_s
            pick = self._buf[0][1]
            for t, item in self._buf:
                if t > target + 1e-9:
                    break
                pick = item
            out.append(pick)
        return out


# ── camera ──────────────────────────────────────────────────────────────────

def _backend() -> int:
    # AVFoundation is the macOS capture API; Continuity Camera is only visible there.
    return cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY


def list_cameras(max_index: int = 6) -> List[Tuple[int, int, int]]:
    """(index, width, height) for every device index that returns a frame."""
    # Probing an index with no device makes OpenCV log a warning per index;
    # silence that for the probe only.
    can_silence = hasattr(cv2, "setLogLevel") and hasattr(cv2, "getLogLevel")
    previous = cv2.getLogLevel() if can_silence else None
    if can_silence:
        cv2.setLogLevel(0)                   # 0 = silent
    found = []
    try:
        for i in range(max_index):
            cap = cv2.VideoCapture(i, _backend())
            try:
                if cap.isOpened():
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        found.append((i, frame.shape[1], frame.shape[0]))
            finally:
                cap.release()
    finally:
        if can_silence:
            cv2.setLogLevel(previous)
    return found


def open_camera(index: int, width: int = 1920, height: int = 1080) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, _backend())
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera {index}; run with --list to see devices "
                           f"(on macOS, the terminal also needs camera permission)")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # Ask for the smallest driver-side queue, so a slow loop reads the newest frame
    # rather than one captured several iterations ago. Not every backend honours it.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


# ── perception + rendering ──────────────────────────────────────────────────

class LiveOverlay:
    """
    One frame in, perception out, and a rendered overlay.

    Args:
        cfg: parsed configs/demo.yaml (thresholds, seq_len).
        pipeline: a `PerceptionPipeline` (segmentation + BEV).
        detector: the single-frame detector `select_detector` chose, in eval mode.
        assumption: the camera geometry every BEV output rests on.
    """

    def __init__(self, cfg: dict, pipeline, detector, assumption: CameraAssumption,
                 device, spacing_s: float = TRAINING_SPACING_S) -> None:
        self.cfg = cfg
        self.pipeline = pipeline
        self.detector = detector
        self.assumption = assumption
        self.device = device
        # (3, 3) and (4, 4), exactly as process_frame documents.
        self.K = torch.from_numpy(estimate_intrinsics(assumption.hfov_deg))
        self.T = torch.from_numpy(assumed_extrinsics(assumption))
        self.xb = tuple(pipeline.bev_cfg["xbound"])
        self.yb = tuple(pipeline.bev_cfg["ybound"])
        # Buffers the uint8 FOV-normalised frames (1 MB each), not float tensors (4x).
        self.window: TemporalWindow[np.ndarray] = TemporalWindow(cfg.get("seq_len", 3), spacing_s)

    @torch.no_grad()
    def step(self, frame_rgb: np.ndarray, t_s: float) -> dict:
        """Run perception on one RGB camera frame captured at `t_s` seconds."""
        norm = fov_normalize(frame_rgb, self.assumption.hfov_deg)
        self.window.push(t_s, norm)

        x = normalize_for_model(norm).unsqueeze(0).to(self.device)
        boxes_l, scores_l, labels_l = self.detector(x)
        boxes, scores, labels = boxes_l[0].cpu(), scores_l[0].cpu(), labels_l[0].cpu()
        keep = scores >= self.cfg.get("det_score_threshold", 0.3)

        window = torch.stack([normalize_for_model(f) for f in self.window.sample()])
        out = self.pipeline.process_frame(window, self.K, self.T)
        out["boxes"], out["scores"], out["labels"] = boxes[keep], scores[keep], labels[keep]
        out["norm_rgb"] = norm
        return out

    def render(self, out: dict, fps: Optional[float] = None) -> np.ndarray:
        """
        BGR composite for cv2.imshow: camera + overlays | BEV, above a status strip.

        Drawn in BGR throughout because the visualize palettes are BGR.
        """
        frame = cv2.cvtColor(out["norm_rgb"], cv2.COLOR_RGB2BGR)
        vis = overlay_segmentation(frame, np.asarray(out["seg_mask"]), alpha=0.45,
                                   skip_classes=(1,))
        vis = draw_boxes(vis, out["boxes"].tolist(), out["labels"].tolist(),
                         out["scores"].tolist())
        bev = draw_bev(out["bev_boxes"], out["bev_scores"], out["bev_labels"],
                       self.xb, self.yb, seg=None, canvas_px=vis.shape[0])
        draw_range_bands(bev, self.xb, self.yb)
        panel = np.hstack([vis, bev])
        strip = status_strip(panel.shape[1], self.assumption, fps, self.window.warm,
                             n_det=len(out["boxes"]), n_bev=len(out["bev_boxes"]))
        return np.vstack([panel, strip])


def status_strip(width: int, assumption: CameraAssumption, fps: Optional[float],
                 warm: bool, n_det: int = 0, n_bev: int = 0, height: int = 46) -> np.ndarray:
    """
    Always-on state line. OUTSIDE ODD is unconditional on this path: the camera is
    uncalibrated and there is no trust score yet (A-8), so the strip never claims
    otherwise and never shows a number that is not a measurement.
    """
    critical = (72, 73, 227)            # BGR of #e34948, the reserved critical status colour
    bar = np.full((height, width, 3), 24, dtype=np.uint8)
    cv2.rectangle(bar, (0, 0), (6, height), critical, -1)
    cv2.putText(bar, "OUTSIDE ODD - DO NOT RELY   uncalibrated camera, no trust score yet",
                (14, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 245, 245), 1, cv2.LINE_AA)
    parts = [f"ASSUMED GEOMETRY: {assumption.describe()}",
             f"{n_det} det / {n_bev} BEV"]
    if fps is not None:
        parts.append(f"{fps:.1f} FPS")
    if not warm:
        parts.append("temporal window warming up")
    cv2.putText(bar, _ascii("   |   ".join(parts))[:150], (14, 37), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (170, 170, 170), 1, cv2.LINE_AA)
    return bar


def load_models(cfg_path: str, hfov_deg: float, device):
    """(cfg, pipeline, detector, (detector_name, rationale)) — needs the checkpoints."""
    import yaml

    from demo.pipeline import PerceptionPipeline
    from models.detection.train_detector import load_detector

    cfg = yaml.safe_load(open(cfg_path))
    pipeline = PerceptionPipeline(cfg, device)
    name, det_cfg_path, det_ckpt, why = select_detector(hfov_deg)
    detector = load_detector(yaml.safe_load(open(det_cfg_path)), det_ckpt, device)
    return cfg, pipeline, detector, (name, why)


def run(args) -> None:
    from models.detection.train_detector import _pick_device

    device = _pick_device()
    assumption = CameraAssumption(hfov_deg=args.hfov, height_m=args.mount_height,
                                  pitch_deg=args.pitch)
    cfg, pipeline, detector, (det_name, why) = load_models(args.config, args.hfov, device)
    print(f"device {device} | detector '{det_name}': {why}")
    print(f"{assumption.describe()} | FOV crop keeps {crop_fraction(args.hfov) * 100:.0f}% of the width")
    overlay = LiveOverlay(cfg, pipeline, detector, assumption, device, spacing_s=args.spacing)

    cap = open_camera(args.camera)
    stamps: Deque[float] = deque(maxlen=30)
    latencies: List[float] = []
    t_start = time.monotonic()
    warned_portrait = False
    try:
        while True:
            ok, bgr = cap.read()
            if not ok or bgr is None:
                print("camera returned no frame; stopping")
                break
            t = time.monotonic()
            if not warned_portrait and bgr.shape[0] > bgr.shape[1]:
                print("warning: portrait frames. The models expect a landscape road view; "
                      "rotate the phone.")
                warned_portrait = True
            out = overlay.step(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), t)
            latencies.append(time.monotonic() - t)
            stamps.append(t)
            fps = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 else None
            cv2.imshow(WINDOW_TITLE, overlay.render(out, fps))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if args.max_seconds and t - t_start >= args.max_seconds:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if latencies:
        ms = [1000 * s for s in latencies]
        print(f"{len(ms)} frames | perception {statistics.mean(ms):.0f} ms mean, "
              f"{statistics.median(ms):.0f} ms median | "
              f"{1000 / statistics.mean(ms):.1f} FPS (A-3 target: 10+ end to end)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m demo.live_camera",
                                 description="Live perception overlays from a webcam or iPhone Continuity Camera.")
    ap.add_argument("--list", action="store_true", help="list camera indexes that return frames, then exit")
    ap.add_argument("--camera", type=int, default=0, help="device index (see --list)")
    ap.add_argument("--hfov", type=float, default=70.0,
                    help="horizontal FOV guess in degrees (iPhone main lens ~70); A-4 replaces this")
    ap.add_argument("--mount-height", type=float, default=1.3, help="camera height above the road, metres")
    ap.add_argument("--pitch", type=float, default=0.0, help="degrees, nose-down positive")
    ap.add_argument("--spacing", type=float, default=TRAINING_SPACING_S,
                    help="seconds between temporal-window frames (training rate: 0.5)")
    ap.add_argument("--config", default="configs/demo.yaml")
    ap.add_argument("--max-seconds", type=float, default=0.0, help="stop after this long (0 = until q)")
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.list:
        cams = list_cameras()
        if not cams:
            print("no cameras returned a frame (on macOS, grant the terminal camera access)")
        for i, w, h in cams:
            print(f"  camera {i}: {w}x{h}")
        print("Continuity Camera is usually not index 0 (that is the built-in camera).")
        return
    run(args)


if __name__ == "__main__":
    main()
