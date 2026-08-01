"""
P12-1 — Frame stream player.

Turns a nuScenes scene into a time-ordered stream, so the Phase 12 agent consumes
a *drive* rather than a frozen frame_id.

TWO STREAMS, TWO PURPOSES
-------------------------
    keyframes — 2 Hz, annotated. Everything the agent says is checkable against
        ground truth. Used by evaluation/streaming_eval.py.
    sweeps    — 12 Hz, unannotated. Smooth, realistic playback for the demo.

Availability constrains this (Phase 9 inventory):
    * v1.0-mini HAS camera sweeps (1,938 CAM_FRONT) and holds all three night
      scenes. The live demo runs on mini — exactly right, since the night scenes
      are where the trust layer earns its place.
    * The trainval blob has NO sweeps and is 100% daytime. Keyframes only there.
`FrameStream` asserts the requested mode is available and fails loudly; silently
falling back from 12 Hz to 2 Hz would make the demo choppy for reasons nobody
could diagnose later.

SWEEPS HAVE NO ANNOTATIONS
--------------------------
nuScenes annotates only keyframes, so on sweep frames there is no GT. Perception
runs fine (it needs no labels), but metrics must restrict themselves to keyframes.
Interpolating GT onto sweeps is acceptable for visualisation and not for numbers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Literal, Optional

import numpy as np
import torch

StreamMode = Literal["keyframes", "sweeps"]


@dataclass
class StreamFrame:
    """One frame of the stream."""
    frame_idx: int
    timestamp_us: int              # nuScenes microsecond timestamp; use for real dt
    sd_token: str                  # sample_data token, always present
    filename: str
    is_keyframe: bool
    sample_token: Optional[str]    # keyframes only; None on sweeps
    ego_pose: dict                 # for ego-motion compensation
    image: Optional[torch.Tensor] = None   # (3, H, W) normalised, if loaded


class FrameStream:
    """
    Iterate a scene's frames in time order.

    Args:
        nusc: NuScenes instance.
        data_root: dataset root.
        scene_name: e.g. "scene-1094".
        mode: "keyframes" (2 Hz, annotated) or "sweeps" (12 Hz, unannotated).
        camera: which camera to stream.
        load_images: decode and normalise images (False = metadata only, fast).
        image_size: (H, W) to resize to.

    The `sample_data` table is a linked list per sensor with `next`/`prev` tokens
    and an `is_key_frame` flag, which gives strict time order for free. Filenames
    are NOT chronologically sortable, so walking the list is the only correct way.

    USE REAL TIMESTAMPS, NOT A FIXED dt
    -----------------------------------
    Sweep intervals are not exactly uniform. Anything computing a rate of change —
    TTC, closing speed, velocity — must divide by the actual timestamp delta.
    Assuming a constant 1/12 s puts a systematic error into every TTC the monitor
    reports, and TTC is its most safety-relevant number. `dt_seconds` exposes the
    true delta.
    """

    def __init__(self, nusc, data_root, scene_name: str,
                 mode: StreamMode = "keyframes", camera: str = "CAM_FRONT",
                 load_images: bool = True, image_size=(448, 800)) -> None:
        from pathlib import Path

        if mode not in ("keyframes", "sweeps"):
            raise ValueError(f"mode must be 'keyframes' or 'sweeps', got {mode!r}")
        self.nusc = nusc
        self.data_root = Path(data_root)
        self.scene_name = scene_name
        self.mode = mode
        self.camera = camera
        self.load_images = load_images
        self.image_size = tuple(image_size)

        scene = next((s for s in nusc.scene if s["name"] == scene_name), None)
        if scene is None:
            raise ValueError(f"scene {scene_name!r} not found")

        first = nusc.get("sample", scene["first_sample_token"])
        if camera not in first["data"]:
            raise ValueError(f"camera {camera!r} not in scene {scene_name!r}")

        records: List[dict] = []
        token = first["data"][camera]
        while token:
            sd = nusc.get("sample_data", token)
            records.append(sd)
            token = sd["next"]

        if mode == "keyframes":
            records = [r for r in records if r["is_key_frame"]]
        else:
            # Check the FILES, not the metadata. nuScenes ships the full
            # sample_data table for all 850 scenes regardless of which blobs were
            # downloaded, so sweep records exist in metadata even when
            # `sweeps/CAM_FRONT/` is empty on disk. Trusting the metadata here
            # would defer the failure to image-load time, deep inside a demo loop.
            sweeps = [r for r in records if not r["is_key_frame"]]
            if not sweeps or not (self.data_root / sweeps[0]["filename"]).exists():
                raise RuntimeError(
                    f"scene {scene_name!r} has no camera sweep FILES on disk "
                    f"({len(sweeps)} sweep records in metadata) — the trainval "
                    f"blob ships keyframes only. Use mode='keyframes', or stream "
                    f"a v1.0-mini scene for 12 Hz playback."
                )
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def dt_seconds(self, i: int) -> float:
        """Real elapsed time between frame i-1 and frame i (0.0 for the first)."""
        if i <= 0:
            return 0.0
        return (self.records[i]["timestamp"] - self.records[i - 1]["timestamp"]) / 1e6

    def mean_rate_hz(self) -> float:
        """Measured frame rate — sanity-check that sweeps really are ~12 Hz."""
        if len(self.records) < 2:
            return 0.0
        span = (self.records[-1]["timestamp"] - self.records[0]["timestamp"]) / 1e6
        return (len(self.records) - 1) / span if span > 0 else 0.0

    def _to_frame(self, i: int) -> StreamFrame:
        sd = self.records[i]
        return StreamFrame(
            frame_idx=i,
            timestamp_us=int(sd["timestamp"]),
            sd_token=sd["token"],
            filename=sd["filename"],
            is_keyframe=bool(sd["is_key_frame"]),
            sample_token=sd["sample_token"] if sd["is_key_frame"] else None,
            ego_pose=self.nusc.get("ego_pose", sd["ego_pose_token"]),
            image=self._load_image(sd) if self.load_images else None,
        )

    def _load_image(self, sd) -> torch.Tensor:
        """Decode + resize + ImageNet-normalise, matching the rest of the project."""
        from PIL import Image
        from data.transforms import MEAN, STD

        img = Image.open(self.data_root / sd["filename"]).convert("RGB")
        img = img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        arr = (np.asarray(img, dtype=np.float32) / 255.0 - np.array(MEAN, dtype=np.float32)) \
            / np.array(STD, dtype=np.float32)
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous().float()

    def __iter__(self) -> Iterator[StreamFrame]:
        for i in range(len(self.records)):
            yield self._to_frame(i)

    def seek(self, frame_idx: int) -> StreamFrame:
        """Random access, for the Streamlit scrubber."""
        if not 0 <= frame_idx < len(self.records):
            raise IndexError(f"frame {frame_idx} out of range (0..{len(self.records) - 1})")
        return self._to_frame(frame_idx)
