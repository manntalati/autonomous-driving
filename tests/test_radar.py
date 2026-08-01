"""Tests for P10 — radar loading, rasterization, encoding and fusion."""
import numpy as np
import pytest
import torch

from data.radar_utils import (
    NUM_RADAR_BEV_CHANNELS,
    RADAR_BEV_CHANNELS,
    VALID_INVALID_STATES,
    filter_radar_points,
    rasterize_radar_bev,
)
from models.bev.radar_encoder import CameraRadarFusion, RadarBEVEncoder

XB = (-51.2, 51.2, 0.8)
YB = (-51.2, 51.2, 0.8)
GRID = 128


def raw_point(x=10.0, y=0.0, rcs=10.0, vx=0.0, vy=0.0, invalid=0, ambig=3, pdh=1):
    p = np.zeros(18)
    p[0], p[1], p[2] = x, y, 0.0
    p[5] = rcs
    p[8], p[9] = vx, vy
    p[11] = ambig
    p[14] = invalid
    p[15] = pdh
    return p


class TestFilter:
    def test_keeps_valid(self):
        assert len(filter_radar_points(np.array([raw_point()]))) == 1

    def test_drops_ambiguous(self):
        assert len(filter_radar_points(np.array([raw_point(ambig=1)]))) == 0

    def test_drops_high_false_alarm(self):
        assert len(filter_radar_points(np.array([raw_point(pdh=6)]))) == 0

    def test_keeps_low_rcs_valid_cluster(self):
        """invalid_state 4 is 'valid cluster with low RCS' — the pedestrian
        signature. The devkit default drops it; we must not."""
        assert 4 in VALID_INVALID_STATES
        assert len(filter_radar_points(np.array([raw_point(invalid=4)]))) == 1

    def test_strict_mode_matches_devkit(self):
        pts = np.array([raw_point(invalid=4)])
        assert len(filter_radar_points(pts, strict=True)) == 0

    def test_empty_input(self):
        assert len(filter_radar_points(np.zeros((0, 18)))) == 0

    def test_bad_shape_rejected(self):
        with pytest.raises(ValueError):
            filter_radar_points(np.zeros((3, 6)))


class TestRasterize:
    def test_shape_and_channels(self):
        g = rasterize_radar_bev(np.zeros((0, 6)), XB, YB)
        assert g.shape == (NUM_RADAR_BEV_CHANNELS, GRID, GRID)
        assert len(RADAR_BEV_CHANNELS) == NUM_RADAR_BEV_CHANNELS

    def test_empty_grid_is_zero(self):
        assert rasterize_radar_bev(np.zeros((0, 6)), XB, YB).sum() == 0

    def test_point_lands_in_expected_cell(self):
        # a point at ego origin maps to the grid centre
        pts = np.array([[0.0, 0.0, 0.0, 10.0, 0.0, 0.0]])
        g = rasterize_radar_bev(pts, XB, YB, dilate=0, normalize=False)
        assert g[0, GRID // 2, GRID // 2] == 1.0

    def test_out_of_range_dropped(self):
        pts = np.array([[500.0, 500.0, 0.0, 10.0, 0.0, 0.0]])
        assert rasterize_radar_bev(pts, XB, YB).sum() == 0

    def test_duplicate_cells_accumulate(self):
        """np.add.at must accumulate; plain fancy indexing would keep only the last."""
        pts = np.array([[0.0, 0.0, 0.0, 10.0, 0, 0]] * 3, dtype=float)
        g = rasterize_radar_bev(pts, XB, YB, dilate=0, normalize=False)
        assert g[0, GRID // 2, GRID // 2] == 3.0

    def test_value_channels_are_means_not_sums(self):
        pts = np.array([[0.0, 0.0, 0.0, 10.0, 0, 0],
                        [0.0, 0.0, 0.0, 20.0, 0, 0]], dtype=float)
        g = rasterize_radar_bev(pts, XB, YB, dilate=0, normalize=False)
        assert g[1, GRID // 2, GRID // 2] == pytest.approx(15.0)

    def test_dilation_increases_occupancy(self):
        pts = np.array([[0.0, 0.0, 0.0, 10.0, 0.0, 0.0]])
        a = (rasterize_radar_bev(pts, XB, YB, dilate=0)[0] > 0).sum()
        b = (rasterize_radar_bev(pts, XB, YB, dilate=1)[0] > 0).sum()
        assert a == 1 and b == 9

    def test_speed_channel(self):
        pts = np.array([[0.0, 0.0, 0.0, 0.0, 3.0, 4.0]])
        g = rasterize_radar_bev(pts, XB, YB, dilate=0, normalize=False)
        assert g[4, GRID // 2, GRID // 2] == pytest.approx(5.0)


class TestEncoderAndFusion:
    def test_encoder_preserves_resolution(self):
        enc = RadarBEVEncoder(5, 64)
        out = enc(torch.randn(2, 5, 64, 64))
        assert out.shape == (2, 64, 64, 64)

    def test_encoder_rejects_bad_rank(self):
        with pytest.raises(ValueError):
            RadarBEVEncoder()(torch.randn(5, 64, 64))

    @pytest.mark.parametrize("mode", ["concat", "gated"])
    def test_fusion_shape(self, mode):
        f = CameraRadarFusion(32, mode=mode)
        out = f(torch.randn(2, 32, 16, 16), torch.randn(2, 32, 16, 16))
        assert out.shape == (2, 32, 16, 16)

    def test_gated_exposes_gate_concat_does_not(self):
        cam, rad = torch.randn(1, 8, 4, 4), torch.randn(1, 8, 4, 4)
        g = CameraRadarFusion(8, mode="gated")
        g(cam, rad)
        assert g.last_gate() is not None and 0.0 <= g.mean_gate() <= 1.0
        c = CameraRadarFusion(8, mode="concat")
        c(cam, rad)
        assert c.last_gate() is None and c.mean_gate() is None

    def test_gate_is_detached(self):
        """Instrumentation must not hold the graph alive across an epoch."""
        f = CameraRadarFusion(8, mode="gated")
        f(torch.randn(1, 8, 4, 4, requires_grad=True), torch.randn(1, 8, 4, 4))
        assert not f.last_gate().requires_grad

    def test_mismatched_shapes_rejected(self):
        with pytest.raises(ValueError):
            CameraRadarFusion(8)(torch.randn(1, 8, 4, 4), torch.randn(1, 8, 8, 8))

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError):
            CameraRadarFusion(8, mode="average")

    def test_gradients_flow(self):
        f = CameraRadarFusion(8, mode="gated")
        cam = torch.randn(1, 8, 4, 4, requires_grad=True)
        f(cam, torch.randn(1, 8, 4, 4)).sum().backward()
        assert cam.grad is not None
