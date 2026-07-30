"""
P12-1 — Frame stream player.

Turns a nuScenes scene into a time-ordered stream of frames, so the Phase 12
agent consumes a *drive* rather than a frozen frame_id. This is the difference
between the current one-shot `run_agent` and something that behaves like a
driving system.

TWO STREAMS, TWO PURPOSES (decided in the Phase 12 scoping)
-----------------------------------------------------------
    keyframes — 2 Hz, annotated. Everything the agent says here is checkable
        against ground truth. Use for `evaluation/streaming_eval.py`.
    sweeps    — 12 Hz, unannotated. Smooth, realistic playback for the demo.

Availability constrains this, per the Phase 9 inventory:
    * v1.0-mini HAS camera sweeps (1,938 CAM_FRONT) — and holds all three night
      scenes. The live demo therefore runs on mini, which is exactly right: the
      night scenes are where the trust layer earns its place.
    * The trainval blob has NO sweeps at all (`sweeps/` is empty) and is 100%
      daytime. Keyframes only there.
Assert the requested mode is available for the requested scene and fail loudly.
Silently falling back from 12 Hz to 2 Hz would make the demo look choppy for
reasons nobody could diagnose later.

SWEEPS HAVE NO ANNOTATIONS — implications
-----------------------------------------
nuScenes annotates only keyframes. On sweep frames there is no GT, so:
    * the perception stack still runs fine (it needs no labels);
    * `evaluation/streaming_eval.py` must restrict itself to keyframes;
    * sweep timestamps do not align with keyframe timestamps — interpolating GT
      onto sweeps is possible but introduces label noise. Do not do it for
      metrics; it is acceptable only for visualisation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Literal, Optional

import torch

StreamMode = Literal["keyframes", "sweeps"]


@dataclass
class StreamFrame:
    """One frame of the stream."""
    frame_idx: int
    timestamp_us: int          # nuScenes microsecond timestamp; use for real dt
    image: torch.Tensor        # (3, H, W) normalised
    sample_token: Optional[str]  # keyframes only; None on sweeps
    is_keyframe: bool
    sd_token: str              # sample_data token, always present


class FrameStream:
    """
    Iterate a scene's frames in time order.

    Args:
        nusc: NuScenes instance.
        data_root: dataset root.
        scene_name: e.g. "scene-1094".
        mode: "keyframes" (2 Hz, annotated) or "sweeps" (12 Hz, unannotated).
        camera: which camera to stream (default CAM_FRONT).

    Usage:
        for frame in FrameStream(nusc, root, "scene-1094", mode="sweeps"):
            ...

    IMPLEMENTATION NOTES
    --------------------
    The `sample_data` table is a linked list per sensor: each record has `next`
    and `prev` tokens, and `is_key_frame` marks the annotated ones. To stream:
        1. token = scene's first sample -> data[camera]
        2. walk `next` until it is ""
        3. in "keyframes" mode, skip records where is_key_frame is False
    This gives strict time order for free — do not sort by filename, which is not
    chronologically ordered.

    USE REAL TIMESTAMPS, NOT A FIXED dt
    -----------------------------------
    nuScenes sweep intervals are not exactly uniform. Anything computing a rate of
    change — time-to-collision, closing speed, velocity — must divide by the
    actual `timestamp` delta in microseconds. Assuming a constant 1/12 s will put
    a systematic error into every TTC estimate the agent reports, and TTC is the
    single most safety-relevant number in the whole system.

    Reuse `SceneStore._load_image` for normalisation so the stream and the
    existing MCP tools preprocess identically. Divergent preprocessing between
    demo and eval paths is a classic source of "it works in the demo" bugs.
    """

    def __init__(self, nusc, data_root, scene_name: str,
                 mode: StreamMode = "keyframes", camera: str = "CAM_FRONT") -> None:
        raise NotImplementedError("P12-1")

    def __len__(self) -> int:
        raise NotImplementedError("P12-1")

    def __iter__(self) -> Iterator[StreamFrame]:
        raise NotImplementedError("P12-1")

    def seek(self, frame_idx: int) -> StreamFrame:
        """Random access, for the Streamlit scrubber."""
        raise NotImplementedError("P12-1")
