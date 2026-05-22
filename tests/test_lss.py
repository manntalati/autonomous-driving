"""
Unit tests for models/bev/lss.py — Lift-Splat-Shoot BEV transform.

Verify:
  - create_frustum builds the right (D, Hf, Wf, 3) coordinate grid
  - DepthNet emits a valid depth distribution + a context feature
  - get_geometry unprojects correctly (identity-calibration sanity check)
  - lift / splat / forward produce the expected shapes and are differentiable

Random tensors only — no nuScenes data or GPU.
"""

import pytest
import torch

from models.bev.lss import create_frustum, DepthNet, LiftSplatShoot


def _lss(**kw):
    """Small LiftSplatShoot for fast tests: 4×4 feature grid, D=3, 4×4 BEV grid."""
    defaults = dict(
        in_channels=8, image_size=(64, 64), feat_stride=16,
        xbound=(0.0, 8.0, 2.0), ybound=(-4.0, 4.0, 2.0),
        zbound=(-10.0, 10.0, 20.0), dbound=(4.0, 10.0, 2.0), bev_channels=4,
    )
    defaults.update(kw)
    return LiftSplatShoot(**defaults)


class TestCreateFrustum:

    def test_shape(self):
        """64×96 image, stride 16 → 4×6 grid; dbound (4,10,2) → D=3."""
        f = create_frustum((64, 96), 16, (4.0, 10.0, 2.0))
        assert f.shape == (3, 4, 6, 3)

    def test_depth_channel_values(self):
        f = create_frustum((64, 64), 16, (4.0, 10.0, 2.0))
        assert torch.allclose(f[:, 0, 0, 2], torch.tensor([4.0, 6.0, 8.0]))


class TestDepthNet:

    def test_output_shapes(self):
        depth, ctx = DepthNet(in_channels=8, context_channels=4, depth_bins=3)(torch.randn(2, 8, 4, 4))
        assert depth.shape == (2, 3, 4, 4)
        assert ctx.shape == (2, 4, 4, 4)

    def test_depth_is_a_distribution(self):
        """The depth channels are a softmax — they sum to 1 over the depth axis."""
        depth, _ = DepthNet(8, 4, 3)(torch.randn(2, 8, 4, 4))
        assert torch.allclose(depth.sum(dim=1), torch.ones(2, 4, 4), atol=1e-5)


class TestLiftSplatShoot:

    def test_grid_dims(self):
        lss = _lss()
        assert lss.nx[0].item() == 4 and lss.nx[1].item() == 4

    def test_get_geometry_shape(self):
        lss = _lss()
        geom = lss.get_geometry(torch.eye(3).repeat(2, 1, 1), torch.eye(4).repeat(2, 1, 1))
        assert geom.shape == (2, 3, 4, 4, 3)

    def test_get_geometry_identity_projection(self):
        """With K = I and cam_to_ego = I, the ego point is just (u·d, v·d, d)."""
        lss = _lss()
        geom = lss.get_geometry(torch.eye(3).repeat(1, 1, 1), torch.eye(4).repeat(1, 1, 1))
        frustum = lss.frustum  # (D, Hf, Wf, 3) = (u, v, d)
        assert torch.allclose(geom[0, ..., 2], frustum[..., 2], atol=1e-4)
        assert torch.allclose(geom[0, ..., 0], frustum[..., 0] * frustum[..., 2], atol=1e-3)

    def test_lift_shape(self):
        """lift → (B, bev_channels, D, Hf, Wf)."""
        lifted = _lss().lift(torch.randn(2, 8, 4, 4))
        assert lifted.shape == (2, 4, 3, 4, 4)

    def test_splat_shape(self):
        """splat → (B, bev_channels, nx_x, nx_y)."""
        lss = _lss()
        bev = lss.splat(torch.randn(2, 4, 3, 4, 4), torch.randn(2, 3, 4, 4, 3))
        assert bev.shape == (2, 4, 4, 4)

    def test_forward_shape_single_camera(self):
        """N=1: one camera → one BEV grid."""
        lss = _lss()
        bev = lss(torch.randn(2, 1, 8, 4, 4),
                  torch.eye(3).repeat(2, 1, 1, 1), torch.eye(4).repeat(2, 1, 1, 1))
        assert bev.shape == (2, 4, 4, 4)

    def test_forward_shape_multi_camera(self):
        """N=3 cameras all splat into the SAME (B, bev_channels, X, Y) grid."""
        lss = _lss()
        bev = lss(torch.randn(2, 3, 8, 4, 4),
                  torch.eye(3).repeat(2, 3, 1, 1), torch.eye(4).repeat(2, 3, 1, 1))
        assert bev.shape == (2, 4, 4, 4)

    def test_differentiable(self):
        lss = _lss()
        feats = torch.randn(1, 2, 8, 4, 4, requires_grad=True)   # B=1, N=2 cameras
        lss(feats, torch.eye(3).repeat(1, 2, 1, 1), torch.eye(4).repeat(1, 2, 1, 1)).sum().backward()
        assert feats.grad is not None and torch.isfinite(feats.grad).all()
