"""
P10-4 — Radar ablation: does radar close the *night* gap specifically?

THE EXPERIMENT
--------------
This is a 2x2. Two models (camera-only, camera+radar) crossed with two conditions
(day, night), all evaluated on unseen scenes:

                          unseen day        unseen night
    camera only            A                 B
    camera + radar         C                 D

    Day benefit   = C - A
    Night benefit = D - B

The Phase 10 claim is `Night benefit >> Day benefit`. That is the whole point:
if radar merely adds capacity, both improve about equally and the ODD argument
fails. Report both numbers side by side and let the gap speak.

Reuse the Phase 9 condition cells from `evaluation/day_night_audit.py` so the two
phases are directly comparable — same scenes, same frames, same metric.

RANGE STRATIFICATION (the second half of P10-4)
-----------------------------------------------
Report each cell split by ground-truth range from the ego vehicle:

    near   0-20 m     camera depth is reasonably reliable here
    mid    20-35 m
    far    35-51.2 m  camera depth degrades badly; radar measures range directly

The expected shape of the result is that radar's benefit grows with range, and
grows most at long range at night. Averaging over all ranges would blur exactly
the effect worth showing. If the stratified result does NOT show that pattern,
say so — it would suggest the fusion is acting as a generic regulariser rather
than supplying range information.

WATCH OUT
---------
The far bucket will hold few objects at night (fewer annotated boxes, and BEV
detection is hard there), so per-bucket AP is noisy. Report the object count in
every bucket next to its AP. An AP computed over 11 boxes is not a measurement,
and presenting it as one undermines the rest of the analysis.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

RANGE_BUCKETS: List[Tuple[str, float, float]] = [
    ("near", 0.0, 20.0),
    ("mid", 20.0, 35.0),
    ("far", 35.0, 51.2),
]


def bucket_by_range(boxes, labels, scores=None):
    """
    Split BEV boxes into RANGE_BUCKETS by their distance from the ego origin.

    Args:
        boxes: (M, 5) [x, y, length, width, yaw] ego-frame BEV boxes.
        labels: (M,) class ids.
        scores: (M,) optional confidence scores (predictions only).

    Returns:
        dict mapping bucket name -> (boxes, labels, scores) subset.

    Range is `sqrt(x^2 + y^2)` from the ego origin — the same convention the MCP
    tools already use for `range_m`, so the numbers here are directly comparable
    to what the agent reports.
    """
    raise NotImplementedError("P10-4")


def evaluate_cell(model, loader, device, num_classes: int, stratify: bool = True) -> Dict:
    """
    Evaluate one model on one condition cell, optionally stratified by range.

    Returns a dict with overall mAP/AP plus, when stratify=True, per-bucket mAP,
    per-bucket AP, and per-bucket GT object counts.

    Use the BEV detection decoder (`decode_bev_detections`) and match predictions
    to GT in BEV. Note this needs a BEV-specific matcher: `compute_map` in
    models/detection/map.py assumes axis-aligned 2D image boxes and pairwise IoU
    from `compute_iou`, neither of which handles rotated BEV boxes. Two options —
    pick one and state it:
      (a) centre-distance matching at a fixed threshold (e.g. 2 m), which is what
          the official nuScenes detection metric does; simplest and defensible.
      (b) rotated-box IoU, which is more familiar but needs a polygon intersection
          routine (shapely, or torchvision's box_iou_rotated if available).
    Option (a) is recommended — it is the published nuScenes convention, it
    degrades gracefully for the small/rare classes, and it avoids writing a
    rotated-IoU kernel that Phase 11 does not need.
    """
    raise NotImplementedError("P10-4")


def main():
    """
    CLI: run the full 2x2 and print the ablation table.

        python -m evaluation.radar_ablation \
            --camera-ckpt checkpoints/bev_surround_best.pt \
            --radar-ckpt  checkpoints/bev_radar_best.pt

    Print in the same style as day_night_audit.py, then the two headline numbers:

        day benefit  : +0.0XX
        night benefit: +0.0XX      <-- the claim

    Also dump per-frame mean gate values (when the model was trained with
    fusion_mode: gated) grouped by day/night, so the gate-shift figure can be
    plotted straight from the JSON.
    """
    raise NotImplementedError("P10-4")


if __name__ == "__main__":
    main()
