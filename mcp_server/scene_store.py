"""
Frame loader and perception cache for the MCP server.

SceneStore owns the nuScenes-to-tensor conversion and holds all loaded frames
in an in-process dict keyed by UUID frame_ids.  The models are NOT loaded here
— call ModelRegistry.run_perception(frame_id) to run the pipeline and cache the
result back into the store.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes

from data.transforms import get_val_transforms

SEG_CLASS_NAMES     = ["background", "drivable", "lane", "ped_crossing", "walkway"]
DETECT_CLASS_NAMES  = ["car", "pedestrian", "cyclist"]


@dataclass
class FrameRecord:
    """All data associated with one loaded keyframe."""
    frame_id:     str
    scene_name:   str
    frame_idx:    int
    timestamp:    int
    frame_window: torch.Tensor          # (T, 3, H, W)  — last T CAM_FRONT frames
    intrinsic:    torch.Tensor          # (3, 3)
    cam_to_ego:   torch.Tensor          # (4, 4)
    perception:   Optional[dict] = field(default=None)  # cached pipeline output


class SceneStore:
    """
    Loads nuScenes CAM_FRONT frames on demand and caches them by UUID frame_id.

    Usage:
        store = SceneStore(nusc, data_root, seq_len=3)
        rec   = store.load_frame("scene-0103", 5)
        # rec.frame_id is now valid for all tool calls
    """

    def __init__(self, nusc: NuScenes, data_root: str, seq_len: int = 3) -> None:
        self.nusc      = nusc
        self.data_root = Path(data_root)
        self.seq_len   = seq_len
        self._tf       = get_val_transforms()
        self._cache:   dict[str, FrameRecord] = {}

        # Pre-index: scene name → ordered list of sample tokens
        self._scene_samples: dict[str, list[str]] = {}
        for scene in nusc.scene:
            tokens: list[str] = []
            token = scene["first_sample_token"]
            while token:
                tokens.append(token)
                token = nusc.get("sample", token)["next"]
            self._scene_samples[scene["name"]] = tokens

    # ── public API ─────────────────────────────────────────────────────────

    def list_scenes(self) -> list[dict]:
        """Return metadata for every scene available in the dataset."""
        return [
            {
                "name":        s["name"],
                "description": s["description"],
                "num_frames":  len(self._scene_samples[s["name"]]),
            }
            for s in self.nusc.scene
        ]

    def load_frame(self, scene_name: str, frame_idx: int) -> FrameRecord:
        """
        Load a single CAM_FRONT keyframe (plus its temporal window) and cache it.
        Returns the cached FrameRecord if the same (scene_name, frame_idx) was
        already loaded.

        Args:
            scene_name: nuScenes scene name, e.g. "scene-0103".
            frame_idx:  0-based keyframe index within the scene.
        """
        # Return cached record if available
        for rec in self._cache.values():
            if rec.scene_name == scene_name and rec.frame_idx == frame_idx:
                return rec

        tokens = self._scene_samples.get(scene_name)
        if tokens is None:
            raise ValueError(f"Unknown scene: {scene_name!r}")
        if not (0 <= frame_idx < len(tokens)):
            raise ValueError(
                f"frame_idx {frame_idx} out of range [0, {len(tokens) - 1}]"
            )

        frame_window        = self._build_window(tokens, frame_idx)
        intrinsic, cam_to_ego = self._get_calibration(tokens[frame_idx])
        sample              = self.nusc.get("sample", tokens[frame_idx])

        rec = FrameRecord(
            frame_id     = str(uuid.uuid4()),
            scene_name   = scene_name,
            frame_idx    = frame_idx,
            timestamp    = sample["timestamp"],
            frame_window = frame_window,
            intrinsic    = intrinsic,
            cam_to_ego   = cam_to_ego,
        )
        self._cache[rec.frame_id] = rec
        return rec

    def get(self, frame_id: str) -> FrameRecord:
        """Retrieve a cached FrameRecord by its UUID. Raises ValueError if not found."""
        if frame_id not in self._cache:
            raise ValueError(
                f"frame_id {frame_id!r} not found. Call load_frame first."
            )
        return self._cache[frame_id]

    # ── internals ──────────────────────────────────────────────────────────

    def _load_image(self, sample_token: str) -> torch.Tensor:
        """Load, resize, and normalise one CAM_FRONT frame → (3, H, W) tensor."""
        sample = self.nusc.get("sample", sample_token)
        sd     = self.nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        img    = np.array(Image.open(self.data_root / sd["filename"]).convert("RGB"))
        # get_val_transforms has bbox_params, so pass empty bbox args
        return self._tf(image=img, bboxes=[], labels=[])["image"]

    def _build_window(self, tokens: list[str], frame_idx: int) -> torch.Tensor:
        """Return a (seq_len, 3, H, W) window ending at frame_idx, padding start if needed."""
        start  = max(0, frame_idx - self.seq_len + 1)
        window = tokens[start : frame_idx + 1]
        while len(window) < self.seq_len:          # pad by repeating first frame
            window = [window[0]] + window
        return torch.stack([self._load_image(t) for t in window])

    def _get_calibration(self, sample_token: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (intrinsic K (3,3), cam_to_ego T (4,4)) for CAM_FRONT."""
        sample = self.nusc.get("sample", sample_token)
        sd     = self.nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        cs     = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])

        K    = torch.tensor(cs["camera_intrinsic"], dtype=torch.float32)
        rot  = torch.tensor(
            Quaternion(cs["rotation"]).rotation_matrix, dtype=torch.float32
        )
        trans = torch.tensor(cs["translation"], dtype=torch.float32)
        T     = torch.eye(4, dtype=torch.float32)
        T[:3, :3] = rot
        T[:3, 3]  = trans
        return K, T
