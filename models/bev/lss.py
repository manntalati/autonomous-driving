"""
Lift-Splat-Shoot BEV transform (Phase 5).

The idea (Philion & Fidler, ECCV 2020):
  - LIFT  — every feature-map pixel predicts a categorical distribution over D
            discrete depths. The pixel's context feature is "lifted" into a
            frustum: D copies, each weighted by that depth's probability.
  - SPLAT — each frustum point has a known 3D location (from the pixel ray +
            depth + camera calibration). Points are pooled into a top-down BEV
            grid by summing every feature that falls in each cell.

create_frustum is given (a fixed coordinate grid). The exercise is the
projection math in get_geometry and the lift/splat in LiftSplatShoot.
"""
from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn


def create_frustum(image_size: Tuple[int, int], feat_stride: int, dbound: Tuple[float, float, float]) -> torch.Tensor:
    """
    Build the (D, Hf, Wf, 3) frustum: for every feature cell and every depth
    bin, the (u, v, depth) triple — u, v in IMAGE pixel coords, depth in metres.
    Args: image_size — (H, W); feat_stride — backbone downsample factor;
          dbound — (depth_min, depth_max, depth_step).
    Returns: (D, Hf, Wf, 3) tensor, last dim = (u_px, v_px, depth).
    """
    H, W = image_size
    Hf, Wf = H // feat_stride, W // feat_stride
    depths = torch.arange(dbound[0], dbound[1], dbound[2], dtype=torch.float32)
    D = depths.shape[0]
    ds = depths.view(D, 1, 1).expand(D, Hf, Wf)
    us = torch.linspace(0, W - 1, Wf).view(1, 1, Wf).expand(D, Hf, Wf)
    vs = torch.linspace(0, H - 1, Hf).view(1, Hf, 1).expand(D, Hf, Wf)
    return torch.stack([us, vs, ds], dim=-1)


class DepthNet(nn.Module):
    """
    Predicts, per feature-map pixel, a depth distribution AND a context feature.
    A 1×1 conv emits (D + context_channels) channels; the first D are softmaxed
    over depth, the rest are the feature carried through the lift.
    """

    def __init__(self, in_channels: int, context_channels: int, depth_bins: int) -> None:
        """
        Args: in_channels — backbone feature channels; context_channels — width of
              the feature carried into BEV; depth_bins — number of depth bins D.
        """
        super().__init__()
        self.depth_bins = depth_bins
        self.context_channels = context_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, depth_bins + context_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args: x — (B, in_channels, Hf, Wf) backbone feature map.
        Returns: (depth, context) —
          depth   (B, D, Hf, Wf) — softmax over the D depth bins (dim=1).
          context (B, context_channels, Hf, Wf).
        """
        out = self.net(x)
        depth = torch.softmax(out[:, :self.depth_bins], dim=1)
        context = out[:, self.depth_bins:]
        return depth, context


class LiftSplatShoot(nn.Module):
    """
    Full LSS transform: backbone feature map → top-down BEV feature grid.
    """

    def __init__(self, in_channels: int, image_size: Tuple[int, int], feat_stride: int, xbound: Tuple[float, float, float], ybound: Tuple[float, float, float], zbound: Tuple[float, float, float], dbound: Tuple[float, float, float], bev_channels: int = 64) -> None:
        """
        Args:
          in_channels — backbone feature channels feeding DepthNet.
          image_size/feat_stride — fix the frustum size.
          xbound/ybound/zbound — BEV grid [lower, upper, cell_size] per axis.
          dbound — depth bins [min, max, step].
          bev_channels — context-feature width carried through lift→splat.
        """
        super().__init__()
        frustum = create_frustum(image_size, feat_stride, dbound)
        self.register_buffer("frustum", frustum, persistent=False)
        self.depth_bins = frustum.shape[0]
        self.bev_channels = bev_channels
        self.depth_net = DepthNet(in_channels, bev_channels, self.depth_bins)

        # BEV grid geometry. nx = cell count, dx = cell size, bx = first cell centre.
        bounds = torch.tensor([xbound, ybound, zbound], dtype=torch.float32)
        self.register_buffer("bx", bounds[:, 0] + bounds[:, 2] / 2.0, persistent=False)
        self.register_buffer("dx", bounds[:, 2], persistent=False)
        self.register_buffer(
            "nx",
            ((bounds[:, 1] - bounds[:, 0]) / bounds[:, 2]).long(),
            persistent=False,
        )

    def get_geometry(self, intrinsics: torch.Tensor, cam_to_ego: torch.Tensor) -> torch.Tensor:
        """
        Unproject the frustum (u, v, depth) points into 3D ego-frame coordinates.
        Args:
          intrinsics — (B, 3, 3) camera K.
          cam_to_ego — (B, 4, 4) homogeneous camera→ego transform.
        Returns: (B, D, Hf, Wf, 3) — each frustum point's (x, y, z) in the ego frame.
        """
        B = intrinsics.shape[0]
        D, Hf, Wf, _ = self.frustum.shape
        frustum = self.frustum.to(intrinsics.device)

        points = torch.cat(
            [frustum[..., :2] * frustum[..., 2:3], frustum[..., 2:3]], dim=-1
        )
        points = points.view(1, D, Hf, Wf, 3, 1).expand(B, D, Hf, Wf, 3, 1)

        # K⁻¹ → camera-frame 3D point
        k_inv = torch.inverse(intrinsics).view(B, 1, 1, 1, 3, 3)
        cam_points = k_inv @ points

        # camera → ego (rotation + translation)
        rot = cam_to_ego[:, :3, :3].view(B, 1, 1, 1, 3, 3)
        trans = cam_to_ego[:, :3, 3].view(B, 1, 1, 1, 3, 1)
        ego_points = rot @ cam_points + trans
        return ego_points.squeeze(-1)

    def lift(self, features: torch.Tensor) -> torch.Tensor:
        """
        Lift the feature map into a depth-weighted frustum of features.
        Args: features — (B, in_channels, Hf, Wf).
        Returns: (B, bev_channels, D, Hf, Wf).
        """
        depth, context = self.depth_net(features)
        return context.unsqueeze(2) * depth.unsqueeze(1)

    def splat(self, lifted: torch.Tensor, geometry: torch.Tensor, num_cams: int = 1) -> torch.Tensor:
        """
        Pool the lifted frustum features into the BEV grid.
        Args:
          lifted   — (B·N, bev_channels, D, Hf, Wf) depth-weighted features.
          geometry — (B·N, D, Hf, Wf, 3) ego-frame coordinates of each point.
          num_cams — N: cameras per sample; their frustums all sum into one grid.
        Returns: (B, bev_channels, nx_x, nx_y) BEV feature grid.
        """
        BN, C, D, Hf, Wf = lifted.shape
        B = BN // num_cams
        device = lifted.device
        nx_x, nx_y = int(self.nx[0]), int(self.nx[1])

        # ego (x, y, z) → integer BEV cell index
        idx = ((geometry - (self.bx - self.dx / 2.0)) / self.dx).long()

        # flatten the frustum points; each point's sample = (B·N index) // N
        n = BN * D * Hf * Wf
        feats = lifted.permute(0, 2, 3, 4, 1).reshape(n, C)
        idx = idx.reshape(n, 3)
        bn_idx = torch.arange(BN, device=device).view(BN, 1, 1, 1).expand(BN, D, Hf, Wf).reshape(n)
        batch = bn_idx // num_cams

        # keep points whose (x, y) cell is inside the grid
        keep = (
            (idx[:, 0] >= 0) & (idx[:, 0] < nx_x)
            & (idx[:, 1] >= 0) & (idx[:, 1] < nx_y)
        )
        feats, idx, batch = feats[keep], idx[keep], batch[keep]

        # sum-pool into the BEV grid via a flat (B·X·Y) index
        flat = batch * (nx_x * nx_y) + idx[:, 0] * nx_y + idx[:, 1]
        bev = torch.zeros(B * nx_x * nx_y, C, device=device, dtype=feats.dtype)
        bev.index_add_(0, flat, feats)
        return bev.view(B, nx_x, nx_y, C).permute(0, 3, 1, 2).contiguous()

    def forward(self, features: torch.Tensor, intrinsics: torch.Tensor, cam_to_ego: torch.Tensor) -> torch.Tensor:
        """
        Args: features — (B, N, in_channels, Hf, Wf); intrinsics — (B, N, 3, 3);
              cam_to_ego — (B, N, 4, 4). N = number of cameras (1 = single-camera).
        Returns: (B, bev_channels, nx_x, nx_y) — all N cameras of a sample
        splatted into one shared BEV grid.
        """
        _, N = features.shape[:2]
        # flatten cameras into the batch — get_geometry / lift are per-image ops
        geometry = self.get_geometry(intrinsics.flatten(0, 1), cam_to_ego.flatten(0, 1))
        lifted = self.lift(features.flatten(0, 1))
        return self.splat(lifted, geometry, num_cams=N)
