"""Tests for P13 — bring-your-own-video geometry and foreign-camera simulation."""
import math

import numpy as np
import pytest
import torch

from data.foreign_camera import (PRESETS, ForeignCamera, fov_transform, simulate,
                                 transform_boxes)
from demo.byo_video import (FOV_PRESETS, NUSCENES_HFOV_DEG, CameraAssumption,
                            adapt_batchnorm, assumed_extrinsics, backbone_of,
                            crop_fraction, estimate_intrinsics, fov_normalize,
                            normalize_for_model)


class TestCropFraction:
    def test_no_crop_at_or_below_training_fov(self):
        assert crop_fraction(NUSCENES_HFOV_DEG) == 1.0
        assert crop_fraction(50.0) == 1.0

    def test_wider_fov_crops_more(self):
        assert crop_fraction(70) > crop_fraction(100) > crop_fraction(140)

    def test_known_values(self):
        # tan(32.4) / tan(60) for a 120 deg dashcam
        assert crop_fraction(120.0) == pytest.approx(0.366, abs=0.005)
        assert crop_fraction(100.0) == pytest.approx(0.533, abs=0.005)

    def test_matches_closed_form(self):
        for fov in (80.0, 110.0, 150.0):
            expect = math.tan(math.radians(NUSCENES_HFOV_DEG / 2)) / math.tan(math.radians(fov / 2))
            assert crop_fraction(fov) == pytest.approx(expect)


class TestFovNormalize:
    @pytest.mark.parametrize("h,w", [(1080, 1920), (1920, 1080), (2160, 3840), (480, 640)])
    def test_always_outputs_model_input_size(self, h, w):
        frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        assert fov_normalize(frame, 120.0).shape == (448, 800, 3)

    def test_no_crop_when_narrow(self):
        frame = np.random.randint(0, 255, (900, 1600, 3), dtype=np.uint8)
        out = fov_normalize(frame, 60.0)
        assert out.shape == (448, 800, 3)

    def test_crop_is_centred(self):
        """A marker at the centre must survive the crop; corners must not."""
        frame = np.zeros((900, 1600, 3), dtype=np.uint8)
        frame[440:460, 790:810] = 255          # centre marker
        frame[:20, :20] = 128                  # corner marker
        out = fov_normalize(frame, 120.0)
        assert out.max() == 255                # centre kept
        assert not (out == 128).any()          # corner cropped away


class TestIntrinsics:
    def test_implied_fov_is_the_training_fov(self):
        """After normalisation the image always subtends the training FOV,
        whatever the source camera was."""
        for src in (70.0, 120.0, 150.0):
            K = estimate_intrinsics(src)
            implied = 2 * math.degrees(math.atan(400 / K[0, 0]))
            assert implied == pytest.approx(NUSCENES_HFOV_DEG, abs=0.1)

    def test_principal_point_centred(self):
        K = estimate_intrinsics(120.0)
        assert K[0, 2] == pytest.approx(400.0)
        assert K[1, 2] == pytest.approx(224.0)


class TestExtrinsics:
    def test_matches_real_nuscenes_convention(self):
        """cam z -> ego +x, cam x -> ego -y, cam y -> ego -z, from the measured rig."""
        T = assumed_extrinsics(CameraAssumption(pitch_deg=0.0))
        expect = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float32)
        assert np.allclose(T[:3, :3], expect, atol=1e-6)
        assert np.allclose(T[:3, 3], [1.70, 0.0, 1.51], atol=1e-3)

    def test_height_and_pitch_applied(self):
        T = assumed_extrinsics(CameraAssumption(height_m=2.2))
        assert T[2, 3] == pytest.approx(2.2)
        Tp = assumed_extrinsics(CameraAssumption(pitch_deg=6.0))
        assert not np.allclose(T[:3, :3], Tp[:3, :3])

    def test_rotation_stays_orthonormal_under_pitch(self):
        for pitch in (-12.0, 0.0, 9.5):
            R = assumed_extrinsics(CameraAssumption(pitch_deg=pitch))[:3, :3].astype(np.float64)
            assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
            assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-6)

    def test_presets_are_sane(self):
        assert all(30.0 <= v <= 180.0 for v in FOV_PRESETS.values())


class TestNormalizeForModel:
    def test_shape_and_dtype(self):
        t = normalize_for_model(np.random.randint(0, 255, (448, 800, 3), dtype=np.uint8))
        assert t.shape == (3, 448, 800) and t.dtype == torch.float32

    def test_actually_normalised(self):
        grey = np.full((448, 800, 3), 128, dtype=np.uint8)
        t = normalize_for_model(grey)
        assert abs(float(t.mean())) < 1.5      # roughly centred, not raw 0-255


class TestBackboneResolution:
    def test_finds_direct_backbone(self):
        m = torch.nn.Module()
        m.backbone = torch.nn.Conv2d(3, 4, 3)
        assert backbone_of(m) is m.backbone

    def test_finds_backbone_one_wrapper_deep(self):
        """TemporalDetector holds its FPNDetector in .detector; only that has
        .backbone. Returning the wrapper feeds a window-model a single frame."""
        inner = torch.nn.Module()
        inner.backbone = torch.nn.Conv2d(3, 4, 3)
        outer = torch.nn.Module()
        outer.detector = inner
        assert backbone_of(outer) is inner.backbone

    def test_returns_none_when_absent(self):
        assert backbone_of(torch.nn.Linear(2, 2)) is None


class TestAdaptBatchnorm:
    def test_updates_stats_and_restores_eval(self):
        m = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.BatchNorm2d(4))
        m.eval()
        before = m[1].running_mean.clone()
        frames = [normalize_for_model(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
                  for _ in range(3)]
        n = adapt_batchnorm(m, frames, torch.device("cpu"))
        assert n == 1
        assert not torch.allclose(before, m[1].running_mean)
        assert not m[1].training          # must not be left stochastic

    def test_no_frames_is_a_noop(self):
        m = torch.nn.Sequential(torch.nn.BatchNorm2d(3))
        assert adapt_batchnorm(m, [], torch.device("cpu")) == 0

    def test_restores_eval_even_when_forward_fails(self):
        class Boom(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                raise RuntimeError("boom")

        m = Boom()
        m.eval()
        with pytest.raises(RuntimeError):
            adapt_batchnorm(m, [torch.randn(3, 8, 8)], torch.device("cpu"))
        assert not m.bn.training


class TestForeignCamera:
    def test_simulate_preserves_shape_and_dtype(self):
        img = np.random.randint(0, 255, (900, 1600, 3), dtype=np.uint8)
        for cam in PRESETS.values():
            out = simulate(img, cam)
            assert out.shape == img.shape and out.dtype == np.uint8

    def test_simulate_is_deterministic_for_a_seed(self):
        img = np.random.randint(0, 255, (256, 448, 3), dtype=np.uint8)
        a = simulate(img, PRESETS["dashcam"], np.random.default_rng(7))
        b = simulate(img, PRESETS["dashcam"], np.random.default_rng(7))
        assert np.array_equal(a, b)

    def test_wider_camera_degrades_more(self):
        """A harsher preset should move the image further from the original."""
        img = np.random.randint(0, 255, (256, 448, 3), dtype=np.uint8)
        d = {n: float(np.abs(simulate(img, c, np.random.default_rng(0)).astype(float)
                             - img.astype(float)).mean())
             for n, c in PRESETS.items()}
        assert d["phone"] < d["dashcam"] < d["action_cam"]

    def test_fov_transform_identity_when_not_wider(self):
        assert fov_transform(1600, 900, 60.0) == (1.0, 0, 0)

    def test_boxes_move_with_pixels(self):
        """The benchmark is meaningless unless labels follow the content — a
        perfect detector would otherwise score ~0."""
        s, x0, y0 = fov_transform(1600, 900, 110.0)
        assert s < 1.0
        moved = transform_boxes([[0.0, 0.0, 100.0, 100.0]], 1600, 900, 110.0)
        assert moved[0][0] == pytest.approx(x0)
        assert moved[0][2] == pytest.approx(100 * s + x0)

    def test_image_centre_is_a_fixed_point(self):
        s, x0, y0 = fov_transform(1600, 900, 110.0)
        cx, cy = 800.0, 450.0
        assert cx * s + x0 == pytest.approx(cx, abs=1.0)
        assert cy * s + y0 == pytest.approx(cy, abs=1.0)

    def test_boxes_unchanged_when_no_fov_shift(self):
        boxes = [[10.0, 20.0, 30.0, 40.0]]
        assert transform_boxes(boxes, 1600, 900, 60.0) == boxes


class TestRobustTransforms:
    def test_outputs_model_input_and_keeps_boxes(self):
        from data.transforms import get_robust_train_transforms
        t = get_robust_train_transforms()
        img = np.random.randint(0, 255, (900, 1600, 3), dtype=np.uint8)
        out = t(image=img, bboxes=[[100.0, 100.0, 500.0, 600.0]], labels=[0])
        assert out["image"].shape == (3, 448, 800)
        assert len(out["bboxes"]) <= 1        # may be dropped by min_visibility

    def test_boxes_stay_inside_the_frame(self):
        from data.transforms import get_robust_train_transforms
        t = get_robust_train_transforms()
        img = np.random.randint(0, 255, (900, 1600, 3), dtype=np.uint8)
        for _ in range(15):
            out = t(image=img, bboxes=[[50.0, 50.0, 700.0, 700.0]], labels=[0])
            for b in out["bboxes"]:
                assert 0 <= b[0] <= 800 and 0 <= b[2] <= 800
                assert 0 <= b[1] <= 448 and 0 <= b[3] <= 448
