"""Tests for A-1 — live camera v0 (no camera or checkpoints needed)."""
import numpy as np
import pytest
import torch

from demo.byo_video import CameraAssumption
from demo.live_camera import LiveOverlay, TemporalWindow, build_parser, status_strip


class TestTemporalWindow:
    def _fill(self, fps=30.0, seconds=3.0, **kw):
        w = TemporalWindow(**kw)
        n = int(fps * seconds)
        for i in range(n):
            w.push(i / fps, i)
        return w, n

    def test_samples_at_training_spacing_not_consecutive_frames(self):
        w, n = self._fill(fps=30.0, seconds=3.0)
        last = n - 1
        # 0.5 s at 30 fps is 15 frames: the window is [t-1.0, t-0.5, t], not [t-2, t-1, t].
        assert w.sample() == [last - 30, last - 15, last]

    def test_slow_arrival_still_spaced_by_time(self):
        # A 10 FPS loop (the realistic live rate) must still yield ~0.5 s spacing.
        w, n = self._fill(fps=10.0, seconds=3.0)
        last = n - 1
        assert w.sample() == [last - 10, last - 5, last]

    def test_warm_up_repeats_oldest_frame(self):
        w = TemporalWindow(seq_len=3, spacing_s=0.5)
        w.push(0.0, "a")
        assert w.sample() == ["a", "a", "a"] and not w.warm
        w.push(0.6, "b")
        assert w.sample() == ["a", "a", "b"] and not w.warm
        w.push(1.1, "c")
        assert w.sample() == ["a", "b", "c"] and w.warm

    def test_buffer_is_bounded_by_span(self):
        w, _ = self._fill(fps=30.0, seconds=60.0)
        assert len(w) <= int(30.0 * w.span_s) + 2

    def test_rejects_time_going_backwards(self):
        w = TemporalWindow()
        w.push(1.0, 0)
        with pytest.raises(ValueError):
            w.push(0.5, 1)

    def test_empty_sample_raises(self):
        with pytest.raises(ValueError):
            TemporalWindow().sample()


class _StubDetector:
    def __call__(self, x):
        assert x.shape == (1, 3, 448, 800)
        boxes = torch.tensor([[10.0, 10.0, 60.0, 60.0], [0.0, 0.0, 5.0, 5.0]])
        return [boxes], [torch.tensor([0.9, 0.1])], [torch.tensor([0, 1])]


class _StubPipeline:
    bev_cfg = {"xbound": [-51.2, 51.2, 0.8], "ybound": [-51.2, 51.2, 0.8]}

    def __init__(self):
        self.windows = []

    def process_frame(self, window, intrinsic, cam_to_ego):
        assert intrinsic.shape == (3, 3) and cam_to_ego.shape == (4, 4)
        self.windows.append(window)
        return {"boxes": None, "scores": None, "labels": None,
                "seg_mask": torch.zeros(448, 800, dtype=torch.long),
                "bev_boxes": np.array([[10.0, 0.0, 4.0, 2.0, 0.0]]),
                "bev_scores": np.array([0.8]), "bev_labels": np.array([0])}


class TestLiveOverlay:
    def _overlay(self):
        pipe = _StubPipeline()
        ov = LiveOverlay({"seq_len": 3, "det_score_threshold": 0.3}, pipe, _StubDetector(),
                         CameraAssumption(hfov_deg=70.0, height_m=1.3), torch.device("cpu"))
        return ov, pipe

    def test_step_filters_scores_and_feeds_a_spaced_window(self):
        ov, pipe = self._overlay()
        for i in range(31):                                # 1.0 s at 30 fps
            frame = np.full((1080, 1920, 3), i, dtype=np.uint8)
            out = ov.step(frame, i / 30.0)
        assert out["scores"].tolist() == pytest.approx([0.9])
        window = pipe.windows[-1]
        assert window.shape == (3, 3, 448, 800)
        # Frames 0, 15 and 30 (0.5 s apart), identified by their fill value.
        from data.transforms import MEAN, STD
        fills = [round(float(window[k, 0, 0, 0]) * STD[0] * 255 + MEAN[0] * 255) for k in range(3)]
        assert fills == [0, 15, 30]

    def test_render_shape(self):
        ov, _ = self._overlay()
        out = ov.step(np.zeros((1080, 1920, 3), dtype=np.uint8), 0.0)
        img = ov.render(out, fps=12.3)
        assert img.dtype == np.uint8
        assert img.shape == (448 + 46, 800 + 448, 3)

    def test_status_strip_never_claims_in_odd(self):
        strip = status_strip(900, CameraAssumption(), fps=None, warm=True)
        assert strip.shape == (46, 900, 3)
        assert tuple(strip[10, 2]) == (72, 73, 227)        # critical keyline, always


def test_cli_defaults():
    args = build_parser().parse_args([])
    assert args.spacing == 0.5 and args.hfov == 70.0 and not args.list
