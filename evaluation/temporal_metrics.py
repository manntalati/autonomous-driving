"""
Temporal consistency metric (Phase 6, ticket P6-4).

A single-frame detector judged only by mAP can still be temporally unstable —
an object detected at t-1 and t+1 but missed at t. That "flicker" is invisible
to mAP but very visible to a human watching the output video.

compute_flicker_rate quantifies it: across a scene's ordered frames, count how
often a GT object that IS detected in both neighbouring frames is MISSED in
the middle one. Lower is better. Pair this with mAP (Phase 2's compute_map):
mAP for accuracy, flicker rate for temporal consistency.
"""
from __future__ import annotations
from typing import List
import numpy as np


def _iou_numpy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU of one box (4,) against many boxes (M, 4), pascal-voc [x1,y1,x2,y2]."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_box = max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0)
    area_boxes = (np.clip(boxes[:, 2] - boxes[:, 0], 0, None)
                  * np.clip(boxes[:, 3] - boxes[:, 1], 0, None))
    union = area_box + area_boxes - inter
    return inter / np.clip(union, 1e-9, None)


def compute_flicker_rate(seq_pred_boxes: List[np.ndarray], seq_gt_boxes: List[np.ndarray], seq_gt_instances: List[List[str]], iou_threshold: float = 0.5) -> float:
    """
    Measure detection flicker across one scene's ordered frames.
    Args:
      seq_pred_boxes — per frame: (P, 4) predicted boxes, in frame order.
      seq_gt_boxes — per frame: (G, 4) GT boxes.
      seq_gt_instances — per frame: list of G instance ids (same id = same
        physical object across frames; from nuScenes 'instance_token').
      iou_threshold — IoU for counting a GT object as "detected".
    Returns: flicker_rate ∈ [0, 1] — of all (object, middle-frame) triples
      where the object is detected in BOTH neighbouring frames, the fraction
      where it is MISSED in the middle frame.
    """
    num_frames = len(seq_gt_boxes)
    detected: List[dict] = []
    for f in range(num_frames):
        preds = np.asarray(seq_pred_boxes[f], dtype=np.float32).reshape(-1, 4)
        gts = np.asarray(seq_gt_boxes[f], dtype=np.float32).reshape(-1, 4)
        instances = seq_gt_instances[f]
        frame_detected: dict = {}
        for g in range(gts.shape[0]):
            if preds.shape[0] == 0:
                frame_detected[instances[g]] = False
            else:
                frame_detected[instances[g]] = bool(
                    _iou_numpy(gts[g], preds).max() >= iou_threshold
                )
        detected.append(frame_detected)

    candidates = 0
    flickers = 0
    for t in range(1, num_frames - 1):
        for inst_id, det_now in detected[t].items():
            det_prev = detected[t - 1].get(inst_id, False)
            det_next = detected[t + 1].get(inst_id, False)
            if det_prev and det_next:
                candidates += 1
                if not det_now:
                    flickers += 1

    return flickers / candidates if candidates > 0 else 0.0
