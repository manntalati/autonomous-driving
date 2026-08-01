"""
Centre-distance matching and mAP for rotated BEV boxes.

WHY NOT REUSE models/detection/map.py (blocker fix)
---------------------------------------------------
`compute_map` there matches with `compute_iou`, which assumes axis-aligned 2-D
image boxes in [x1, y1, x2, y2]. BEV boxes are [x, y, length, width, yaw] —
rotated, in metres, in the ego frame. Feeding them to the image-space matcher
silently produces meaningless overlaps (it would treat `length`/`width` as
absolute corner coordinates), and every AP computed from it would be garbage
that still looks like a plausible number.

Two ways to fix it. We take the first:

  (a) CENTRE-DISTANCE matching at a fixed threshold in metres. This is the
      official nuScenes detection convention (they average AP over thresholds
      {0.5, 1, 2, 4} m). It needs no polygon geometry, degrades gracefully for
      small/rare classes whose extents are poorly estimated, and — importantly
      for Phase 12 — uses the same notion of "range" the MCP tools already
      report, so numbers stay comparable across the project.
  (b) Rotated-box IoU, which is more familiar from 2-D detection but requires a
      polygon-intersection routine and punishes extent errors that centre
      distance forgives. Nothing downstream in Phases 10-12 needs it.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from models.detection.map import compute_ap

# nuScenes averages AP over these centre-distance thresholds (metres).
NUSCENES_DIST_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)


def centre_distance_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Pairwise Euclidean distance between BEV box centres.

    Args:
        boxes_a: (N, >=2) [x, y, ...] — extra columns ignored.
        boxes_b: (M, >=2)
    Returns: (N, M) float64 distances in metres. Empty inputs give a correctly
    shaped empty array rather than raising.
    """
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, np.shape(boxes_a)[-1] if len(boxes_a) else 2)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, np.shape(boxes_b)[-1] if len(boxes_b) else 2)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    diff = a[:, None, :2] - b[None, :, :2]
    return np.sqrt((diff ** 2).sum(axis=-1))


def match_by_centre_distance(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    threshold: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Greedy score-ordered matching of predictions to ground truth.

    Args:
        pred_boxes: (N, >=2) predicted BEV boxes.
        pred_scores: (N,) confidence scores.
        gt_boxes: (M, >=2) ground-truth BEV boxes.
        threshold: max centre distance (metres) for a valid match.

    Returns:
        (tp, matched_gt) where
          tp: (N,) bool, in the ORIGINAL prediction order — True if that
              prediction matched an unused GT within `threshold`.
          matched_gt: (N,) int, index of the matched GT or -1.

    Greedy by descending score is the standard convention: the most confident
    prediction gets first claim on a GT object, and each GT can be matched at
    most once so duplicate detections correctly count as false positives.
    """
    n = len(pred_boxes)
    tp = np.zeros(n, dtype=bool)
    matched_gt = np.full(n, -1, dtype=int)
    if n == 0 or len(gt_boxes) == 0:
        return tp, matched_gt

    dist = centre_distance_matrix(pred_boxes, gt_boxes)
    used = np.zeros(len(gt_boxes), dtype=bool)
    for i in np.argsort(-np.asarray(pred_scores, dtype=np.float64)):
        d = dist[i].copy()
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= threshold:
            tp[i] = True
            matched_gt[i] = j
            used[j] = True
    return tp, matched_gt


def compute_bev_map(
    predictions: List[dict],
    ground_truths: List[dict],
    num_classes: int,
    threshold: float = 2.0,
) -> Tuple[float, List[float]]:
    """
    mAP over a dataset of BEV frames, using centre-distance matching.

    Args:
        predictions: per-frame dicts with "boxes" (N,5), "scores" (N,), "labels" (N,).
        ground_truths: per-frame dicts with "boxes" (M,5), "labels" (M,).
        num_classes: number of detection classes.
        threshold: centre-distance match threshold in metres.

    Returns: (mAP, [AP per class]).

    Mirrors the structure of models/detection/map.compute_map — accumulate
    (score, tp) pairs across the whole dataset per class, sort globally by score,
    build the cumulative PR curve, then integrate with the shared `compute_ap`.
    Sorting globally rather than per frame matters: AP is defined over one ranked
    list for the entire dataset, and per-frame sorting inflates it.
    """
    per_class_ap: List[float] = []
    for c in range(num_classes):
        scores_all: List[float] = []
        tp_all: List[bool] = []
        total_gt = 0

        for pred, gt in zip(predictions, ground_truths):
            p_boxes = np.asarray(pred.get("boxes", []), dtype=np.float64).reshape(-1, 5)
            p_scores = np.asarray(pred.get("scores", []), dtype=np.float64).reshape(-1)
            p_labels = np.asarray(pred.get("labels", []), dtype=np.int64).reshape(-1)
            g_boxes = np.asarray(gt.get("boxes", []), dtype=np.float64).reshape(-1, 5)
            g_labels = np.asarray(gt.get("labels", []), dtype=np.int64).reshape(-1)

            pm = p_labels == c
            gm = g_labels == c
            total_gt += int(gm.sum())
            if pm.sum() == 0:
                continue

            tp, _ = match_by_centre_distance(p_boxes[pm], p_scores[pm], g_boxes[gm], threshold)
            scores_all.extend(p_scores[pm].tolist())
            tp_all.extend(tp.tolist())

        if total_gt == 0 or len(scores_all) == 0:
            # No GT for this class anywhere: AP is undefined. Use 0.0 to match the
            # existing compute_map convention so the two are comparable, but the
            # caller should report per-class GT counts alongside AP.
            per_class_ap.append(0.0)
            continue

        order = np.argsort(-np.asarray(scores_all))
        tp_sorted = np.asarray(tp_all, dtype=np.float64)[order]
        cum_tp = np.cumsum(tp_sorted)
        cum_fp = np.cumsum(1.0 - tp_sorted)
        precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
        recall = cum_tp / total_gt
        per_class_ap.append(compute_ap(precision, recall))

    mean_ap = float(np.mean(per_class_ap)) if per_class_ap else 0.0
    return mean_ap, per_class_ap


def box_ranges(boxes: np.ndarray) -> np.ndarray:
    """
    Euclidean range of each BEV box centre from the ego origin, in metres.
    Matches the `range_m` convention used by the MCP perception tools.
    """
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, np.shape(boxes)[-1] if len(boxes) else 2)
    if len(b) == 0:
        return np.zeros((0,), dtype=np.float64)
    return np.sqrt((b[:, :2] ** 2).sum(axis=1))
