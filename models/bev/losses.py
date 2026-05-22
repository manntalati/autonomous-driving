"""
BEV detection targets + loss (Phase 5).

Centre-based supervision (CenterNet-style):
  - object centres are rendered as 2D Gaussian peaks on a per-class heatmap;
  - box parameters (sub-cell offset, size, heading) are regressed only at the
    exact centre cell of each GT object.
"""
from __future__ import annotations
import math
from typing import Dict, List, Tuple
import torch
import torch.nn as nn


def _draw_gaussian(heatmap: torch.Tensor, center: Tuple[int, int], radius: int) -> None:
    """Render a 2D Gaussian peak into heatmap[ix±r, iy±r] in place (max-combined)."""
    ix, iy = center
    nx, ny = heatmap.shape
    sigma = (2 * radius + 1) / 6.0
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    g = torch.exp(-(coords[:, None] ** 2 + coords[None, :] ** 2) / (2 * sigma * sigma))
    x0, x1 = max(0, ix - radius), min(nx, ix + radius + 1)
    y0, y1 = max(0, iy - radius), min(ny, iy + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return
    gx0, gy0 = x0 - (ix - radius), y0 - (iy - radius)
    gpatch = g[gx0:gx0 + (x1 - x0), gy0:gy0 + (y1 - y0)]
    heatmap[x0:x1, y0:y1] = torch.maximum(heatmap[x0:x1, y0:y1], gpatch)


def _focal_loss(pred: torch.Tensor, gt: torch.Tensor,
                alpha: float = 2.0, beta: float = 4.0, eps: float = 1e-4) -> torch.Tensor:
    """
    Penalty-reduced focal loss (CenterNet) for a Gaussian heatmap target.
    Positive = the exact peak cell (gt == 1); every other cell is a negative
    whose loss is down-weighted by (1 - gt)^beta — cells near a peak count less.
    """
    pred = pred.clamp(eps, 1.0 - eps)
    pos = gt.eq(1.0).float()
    neg = 1.0 - pos
    pos_loss = torch.log(pred) * (1.0 - pred) ** alpha * pos
    neg_loss = torch.log(1.0 - pred) * pred ** alpha * ((1.0 - gt) ** beta) * neg
    n_pos = pos.sum()
    pos_loss, neg_loss = pos_loss.sum(), neg_loss.sum()
    if n_pos == 0:
        return -neg_loss
    return -(pos_loss + neg_loss) / n_pos


def encode_bev_targets(boxes: torch.Tensor, labels: torch.Tensor, num_classes: int, xbound: Tuple[float, float, float], ybound: Tuple[float, float, float]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Turn a frame's GT BEV boxes into dense training targets.
    Args:
      boxes — (N, 5) [x, y, length, width, yaw] ego-frame BEV boxes.
      labels — (N,) class ids.
      num_classes — heatmap channel count.
      xbound/ybound — BEV grid [lower, upper, cell_size] per axis.
    Returns: (heatmap, regression, reg_mask) —
      heatmap    (num_classes, X, Y) — Gaussian peaks at object centres.
      regression (6, X, Y) — [offset_x, offset_y, length, width, sin yaw, cos yaw].
      reg_mask   (X, Y) — 1 at object-centre cells, 0 elsewhere.
    """
    nx = int(round((xbound[1] - xbound[0]) / xbound[2]))
    ny = int(round((ybound[1] - ybound[0]) / ybound[2]))
    heatmap = torch.zeros(num_classes, nx, ny)
    regression = torch.zeros(6, nx, ny)
    reg_mask = torch.zeros(nx, ny)

    for box, label in zip(boxes, labels):
        x, y, length, width, yaw = (float(v) for v in box)
        cx = (x - xbound[0]) / xbound[2]
        cy = (y - ybound[0]) / ybound[2]
        ix, iy = int(cx), int(cy)
        if not (0 <= ix < nx and 0 <= iy < ny):
            continue
        radius = max(1, int(0.5 * min(length / xbound[2], width / ybound[2])))
        _draw_gaussian(heatmap[int(label)], (ix, iy), radius)
        regression[0, ix, iy] = cx - ix
        regression[1, ix, iy] = cy - iy
        regression[2, ix, iy] = length
        regression[3, ix, iy] = width
        regression[4, ix, iy] = math.sin(yaw)
        regression[5, ix, iy] = math.cos(yaw)
        reg_mask[ix, iy] = 1.0

    return heatmap, regression, reg_mask

class BEVDetectionLoss(nn.Module):
    """
    Heatmap loss (penalty-reduced focal, CenterNet-style) + L1 box regression
    evaluated only at GT centre cells.
    """
    def __init__(self, num_classes: int, xbound: Tuple[float, float, float], ybound: Tuple[float, float, float], heatmap_weight: float = 1.0, reg_weight: float = 1.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.xbound = xbound
        self.ybound = ybound
        self.heatmap_weight = heatmap_weight
        self.reg_weight = reg_weight

    def forward(self, pred_heatmap: torch.Tensor, pred_reg: torch.Tensor, targets: List[Dict]) -> Tuple[torch.Tensor, dict]:
        """
        Args:
          pred_heatmap — (B, num_classes, X, Y) predicted centre probabilities.
          pred_reg     — (B, 6, X, Y) predicted box-regression maps.
          targets      — list of B dicts, each with 'boxes' (N,5) and 'labels' (N,).
        Returns: (total_loss, log_dict) with keys 'loss', 'hm_loss', 'reg_loss'.
        """
        device = pred_heatmap.device
        tgt_hm, tgt_reg, tgt_mask = [], [], []
        for t in targets:
            hm, reg, mask = encode_bev_targets(
                t["boxes"], t["labels"], self.num_classes, self.xbound, self.ybound
            )
            tgt_hm.append(hm)
            tgt_reg.append(reg)
            tgt_mask.append(mask)
        tgt_hm = torch.stack(tgt_hm).to(device)
        tgt_reg = torch.stack(tgt_reg).to(device)
        tgt_mask = torch.stack(tgt_mask).to(device)

        hm_loss = _focal_loss(pred_heatmap, tgt_hm)
        # L1 box regression, only at GT centre cells
        num_pos = tgt_mask.sum().clamp(min=1.0)
        reg_loss = (torch.abs(pred_reg - tgt_reg) * tgt_mask.unsqueeze(1)).sum() / num_pos

        total = self.heatmap_weight * hm_loss + self.reg_weight * reg_loss
        log = {"loss": total.item(), "hm_loss": hm_loss.item(), "reg_loss": reg_loss.item()}
        return total, log