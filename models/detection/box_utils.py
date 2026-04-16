import torch
from typing import Tuple
from torchvision.ops import nms as tv_nms


def compute_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """
    Pairwise IoU between two sets of boxes — fully vectorized, device-agnostic.
    Args: boxes_a — (N, 4) in [x1,y1,x2,y2]; boxes_b — (M, 4) in [x1,y1,x2,y2].
    Returns: (N, M) float32 tensor on boxes_a.device.
    """
    boxes_a = boxes_a.float()
    boxes_b = boxes_b.float()
    N, M = boxes_a.shape[0], boxes_b.shape[0]
    if N == 0 or M == 0:
        return torch.zeros((N, M), dtype=torch.float32, device=boxes_a.device)
    # (N, 1, 2) vs (1, M, 2) → (N, M, 2)
    lt = torch.max(boxes_a[:, None, :2], boxes_b[None, :, :2])
    rb = torch.min(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = (rb - lt).clamp_min(0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp_min(1e-6)


def match_anchors_to_gt(anchors: torch.Tensor, gt_boxes: torch.Tensor, iou_threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized best-GT matching. For each anchor returns the index of its best-IoU GT box
    and a 0/1 label (fg if IoU >= threshold else bg).
    Args: anchors — (N, 4); gt_boxes — (M, 4); iou_threshold — foreground cutoff.
    Returns: (matched_gt_indices (N,) long, match_labels (N,) long) on anchors.device.
    """
    N = anchors.shape[0]
    device = anchors.device
    if gt_boxes.numel() == 0:
        return (
            torch.zeros(N, dtype=torch.long, device=device),
            torch.zeros(N, dtype=torch.long, device=device),
        )
    ious = compute_iou(anchors, gt_boxes)          # (N, M)
    best_iou, best_idx = ious.max(dim=1)           # (N,), (N,)
    labels = (best_iou >= iou_threshold).long()
    return best_idx.long(), labels


def encode_boxes(anchors: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    """
    Vectorized GT → (dx, dy, dw, dh) delta encoding.
    Args: anchors — (N, 4); gt_boxes — (N, 4) matched GT box per anchor, both in [x1,y1,x2,y2].
    Returns: (N, 4) float32 deltas on anchors.device.
    """
    anchors = anchors.float()
    gt_boxes = gt_boxes.float()
    cx_a = (anchors[:, 0] + anchors[:, 2]) * 0.5
    cy_a = (anchors[:, 1] + anchors[:, 3]) * 0.5
    w_a  = (anchors[:, 2] - anchors[:, 0]).clamp_min(1e-6)
    h_a  = (anchors[:, 3] - anchors[:, 1]).clamp_min(1e-6)
    cx_g = (gt_boxes[:, 0] + gt_boxes[:, 2]) * 0.5
    cy_g = (gt_boxes[:, 1] + gt_boxes[:, 3]) * 0.5
    w_g  = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp_min(1e-6)
    h_g  = (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp_min(1e-6)
    dx = (cx_g - cx_a) / w_a
    dy = (cy_g - cy_a) / h_a
    dw = torch.log(w_g / w_a)
    dh = torch.log(h_g / h_a)
    return torch.stack([dx, dy, dw, dh], dim=1)


def decode_boxes(anchors: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """
    Vectorized inverse of encode_boxes. Applies predicted deltas to anchors.
    Args: anchors — (N, 4); deltas — (N, 4) predicted [dx,dy,dw,dh].
    Returns: (N, 4) float32 decoded [x1,y1,x2,y2] boxes.
    """
    anchors = anchors.float()
    deltas = deltas.float()
    cx_a = (anchors[:, 0] + anchors[:, 2]) * 0.5
    cy_a = (anchors[:, 1] + anchors[:, 3]) * 0.5
    w_a  = anchors[:, 2] - anchors[:, 0]
    h_a  = anchors[:, 3] - anchors[:, 1]
    cx = deltas[:, 0] * w_a + cx_a
    cy = deltas[:, 1] * h_a + cy_a
    w  = w_a * torch.exp(deltas[:, 2])
    h  = h_a * torch.exp(deltas[:, 3])
    return torch.stack([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5], dim=1)


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5) -> torch.Tensor:
    """
    Non-maximum suppression via torchvision's fused kernel (CUDA/CPU/MPS-fallback).
    Args: boxes — (N, 4); scores — (N,); iou_threshold — suppression cutoff.
    Returns: 1D LongTensor of kept indices, sorted by score descending.
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    return tv_nms(boxes.float(), scores.float(), iou_threshold).long()
