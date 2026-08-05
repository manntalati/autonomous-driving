"""
Contract tests for the demo entry points.

WHY THIS FILE EXISTS
--------------------
The radar A/B panel shipped broken: it passed `(6,3,448,800)` where the model
needed `(1,6,3,448,800)`, and left the camera tensors on CPU while the radar grid
went to the GPU. Both bugs lived in the CALLER. The smoke test that "verified" it
called `process_bev_pair` directly with the batch dimension already added, on CPU
— so it exercised the function and never touched the code that invokes it.

These tests pin the shape/device contracts so a caller that drifts from them fails
here rather than in a browser. The heavyweight end-to-end checks need nuScenes and
trained checkpoints, so they skip cleanly when those are absent (CI has neither).
"""
from pathlib import Path

import pytest
import torch

DATA_ROOT = Path("data/raw/v1.0-mini")
CKPTS = Path("checkpoints")

needs_data = pytest.mark.skipif(
    not (DATA_ROOT / "v1.0-mini").exists(),
    reason="nuScenes mini not present",
)
needs_ckpts = pytest.mark.skipif(
    not (CKPTS / "temporal_best.pt").exists(),
    reason="trained checkpoints not present",
)


class TestProcessFrameContract:
    """These run without data or checkpoints — they only need the guard clauses."""

    def _pipe(self):
        """A PerceptionPipeline shell with the guards but no loaded models."""
        from demo.pipeline import PerceptionPipeline
        return PerceptionPipeline.__new__(PerceptionPipeline)

    def test_rejects_unbatched_frame_window(self):
        p = self._pipe()
        with pytest.raises(ValueError, match="frame_window must be"):
            PerceptionPipelineCall(p, torch.randn(3, 448, 800), torch.eye(3), torch.eye(4))

    def test_rejects_wrong_rank_intrinsic(self):
        """(1,3,3) used to pass silently — view() reshapes by element count."""
        p = self._pipe()
        with pytest.raises(ValueError, match="intrinsic must be"):
            PerceptionPipelineCall(p, torch.randn(3, 3, 448, 800),
                                   torch.eye(3).unsqueeze(0), torch.eye(4))

    def test_rejects_wrong_rank_extrinsic(self):
        p = self._pipe()
        with pytest.raises(ValueError, match="cam_to_ego must be"):
            PerceptionPipelineCall(p, torch.randn(3, 3, 448, 800),
                                   torch.eye(3), torch.eye(4).unsqueeze(0))


def PerceptionPipelineCall(pipe, window, K, c2e):
    """Invoke process_frame's guards without needing loaded models."""
    from demo.pipeline import PerceptionPipeline
    return PerceptionPipeline.process_frame(pipe, window, K, c2e)


class TestProcessBevPairContract:
    def _pipe_with_radar(self):
        from demo.pipeline import PerceptionPipeline
        p = PerceptionPipeline.__new__(PerceptionPipeline)
        p.bev_radar = torch.nn.Linear(1, 1)      # stand-in with .parameters()
        p.bev_ab_camera = torch.nn.Linear(1, 1)
        return p

    def test_rejects_missing_batch_dimension(self):
        """The exact bug: (N,3,H,W) flattens to 18 'channels' inside the model."""
        from demo.pipeline import PerceptionPipeline
        p = self._pipe_with_radar()
        with pytest.raises(ValueError, match="images must be"):
            PerceptionPipeline.process_bev_pair(
                p, torch.randn(6, 3, 448, 800), torch.eye(3).repeat(1, 6, 1, 1),
                torch.eye(4).repeat(1, 6, 1, 1), torch.randn(1, 5, 128, 128))

    def test_rejects_device_mismatch(self):
        from demo.pipeline import PerceptionPipeline
        p = self._pipe_with_radar()
        imgs = torch.randn(1, 6, 3, 8, 8)
        # model params are on CPU; put one input somewhere else via a meta tensor
        other = torch.randn(1, 5, 8, 8, device="meta")
        with pytest.raises(ValueError, match="on meta|move every input"):
            PerceptionPipeline.process_bev_pair(
                p, imgs, torch.eye(3).repeat(1, 6, 1, 1),
                torch.eye(4).repeat(1, 6, 1, 1), other)

    def test_requires_both_ablation_checkpoints(self):
        from demo.pipeline import PerceptionPipeline
        p = PerceptionPipeline.__new__(PerceptionPipeline)
        p.bev_radar = None
        p.bev_ab_camera = None
        with pytest.raises(RuntimeError, match="both ablation checkpoints"):
            PerceptionPipeline.process_bev_pair(
                p, torch.randn(1, 6, 3, 8, 8), torch.eye(3), torch.eye(4), None)


class TestByoCallerShapes:
    """The BYO app builds calibration itself — pin the shapes it produces."""

    def test_estimate_intrinsics_is_3x3(self):
        from demo.byo_video import estimate_intrinsics
        assert estimate_intrinsics(120.0).shape == (3, 3)

    def test_assumed_extrinsics_is_4x4(self):
        from demo.byo_video import CameraAssumption, assumed_extrinsics
        assert assumed_extrinsics(CameraAssumption()).shape == (4, 4)

    def test_shapes_satisfy_process_frame_contract(self):
        """Guards against the app re-introducing an extra batch dimension."""
        from demo.byo_video import (CameraAssumption, assumed_extrinsics,
                                    estimate_intrinsics)
        K = torch.from_numpy(estimate_intrinsics(120.0))
        T = torch.from_numpy(assumed_extrinsics(CameraAssumption()))
        assert K.dim() == 2 and K.shape[-2:] == (3, 3)
        assert T.dim() == 2 and T.shape[-2:] == (4, 4)

    def test_normalize_for_model_needs_unsqueeze_for_batch(self):
        import numpy as np
        from demo.byo_video import normalize_for_model
        t = normalize_for_model(np.zeros((448, 800, 3), dtype=np.uint8))
        assert t.dim() == 3                      # (3, H, W) — caller adds batch
        assert t.unsqueeze(0).shape == (1, 3, 448, 800)


@needs_data
@needs_ckpts
class TestEndToEndPaths:
    """Full caller paths on the real device. Skipped without data/checkpoints."""

    @pytest.fixture(scope="class")
    def rig(self):
        import yaml
        from nuscenes.nuscenes import NuScenes
        from demo.pipeline import PerceptionPipeline
        cfg = yaml.safe_load(open("configs/demo.yaml"))
        dev = torch.device("cpu")            # deterministic; MPS not needed for shapes
        pipe = PerceptionPipeline(cfg, dev)
        nusc = NuScenes(version="v1.0-mini", dataroot=cfg["data_root"], verbose=False)
        scene = next(s for s in nusc.scene if s["name"] == "scene-0103")
        tok = nusc.get("sample", scene["first_sample_token"])["data"]["CAM_FRONT"]
        return cfg, pipe, nusc, tok, dev

    def test_app_main_view(self, rig):
        from demo.app import _calibration, _load_frame, _surround_inputs
        cfg, pipe, nusc, tok, _ = rig
        window = torch.stack([_load_frame(nusc, cfg["data_root"], tok) for _ in range(3)])
        K, c2e = _calibration(nusc, tok)
        cams = pipe.bev_cfg.get("cameras", ["CAM_FRONT"])
        sur = _surround_inputs(nusc, cfg["data_root"], tok, cams) if len(cams) > 1 else None
        out = pipe.process_frame(window, K, c2e, bev_surround=sur)
        assert {"boxes", "seg_mask", "bev_boxes"} <= set(out)

    def test_radar_ab_panel(self, rig):
        from data.radar_utils import radar_bev_for_sample
        from demo.app import _surround_inputs
        cfg, pipe, nusc, tok, dev = rig
        if not pipe.has_radar_arm:
            pytest.skip("radar ablation checkpoints not present")
        rcfg = pipe.bev_radar_cfg
        imgs, K, c2e = _surround_inputs(nusc, cfg["data_root"], tok, rcfg["cameras"])
        sample = nusc.get("sample", nusc.get("sample_data", tok)["sample_token"])
        rg = torch.from_numpy(radar_bev_for_sample(
            nusc, cfg["data_root"], sample,
            tuple(rcfg["xbound"]), tuple(rcfg["ybound"]))).float().unsqueeze(0).to(dev)
        ab = pipe.process_bev_pair(imgs.unsqueeze(0).to(dev), K.unsqueeze(0).to(dev),
                                   c2e.unsqueeze(0).to(dev), rg)
        assert {"camera", "camera_radar"} <= set(ab)
