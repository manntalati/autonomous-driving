"""
Model registry for the MCP server.

ModelRegistry loads all three trained models once at server start-up and
provides a single run_perception(frame_id) entry point that:
  1. Retrieves the frame tensors from the SceneStore.
  2. Runs PerceptionPipeline.process_frame() if not already cached.
  3. Stores the numpy-converted results back in the FrameRecord.

A module-level singleton (_instance) means the first tool call bears the
model-loading cost; every subsequent call is inference-only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
import torch
from nuscenes.nuscenes import NuScenes

from demo.pipeline import PerceptionPipeline
from data.dataset import version_from_data_root
from mcp_server.scene_store import SceneStore

_instance: "ModelRegistry | None" = None


class ModelRegistry:
    """Owns the pipeline, the nuScenes handle, and the per-frame cache."""

    def __init__(self, cfg_path: str | Path) -> None:
        with open(cfg_path) as f:
            self.cfg = yaml.safe_load(f)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps"  if torch.backends.mps.is_available()
            else "cpu"
        )
        self.pipeline = PerceptionPipeline(self.cfg, self.device)
        self.nusc     = NuScenes(
            version  = version_from_data_root(self.cfg["data_root"]),
            dataroot = self.cfg["data_root"],
            verbose  = False,
        )
        self.store = SceneStore(
            nusc      = self.nusc,
            data_root = self.cfg["data_root"],
            seq_len   = self.cfg.get("seq_len", 3),
        )

    def run_perception(self, frame_id: str) -> dict:
        """
        Run all three models on the frame identified by frame_id and return
        the cached numpy result dict.

        Keys in the returned dict:
          boxes      (N, 4)  — 2-D detection boxes [x1,y1,x2,y2] (pixels)
          scores     (N,)    — detection confidence scores
          labels     (N,)    — detection class indices (0=car,1=ped,2=cyc)
          seg_mask   (H, W)  — segmentation argmax (0–4)
          bev_boxes  (M, 5)  — BEV boxes [x,y,length,width,yaw] (metres)
          bev_scores (M,)    — BEV detection scores
          bev_labels (M,)    — BEV class indices
          bev_seg    (X, Y)  — BEV semantic map argmax (0–4)
        """
        rec = self.store.get(frame_id)
        if rec.perception is None:
            result = self.pipeline.process_frame(
                rec.frame_window, rec.intrinsic, rec.cam_to_ego
            )
            rec.perception = {
                "boxes":      result["boxes"].numpy(),
                "scores":     result["scores"].numpy(),
                "labels":     result["labels"].numpy(),
                "seg_mask":   result["seg_mask"].numpy(),
                "bev_boxes":  result["bev_boxes"].numpy(),
                "bev_scores": result["bev_scores"].numpy(),
                "bev_labels": result["bev_labels"].numpy(),
                "bev_seg":    result["bev_seg"].numpy(),
            }
        return rec.perception


def get_registry(cfg_path: str = "configs/agent.yaml") -> ModelRegistry:
    """Return the module-level singleton, initialising it on first call."""
    global _instance
    if _instance is None:
        _instance = ModelRegistry(cfg_path)
    return _instance
