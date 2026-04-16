import numpy as np
from typing import List, Tuple
from box_utils import compute_iou
import torch

def compute_ap(precision: np.ndarray, recall: np.ndarray) -> float:
    """
    Area under the precision-recall curve (11-point interpolation or COCO-style).
    """
    ap = 0
    time_steps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for t in time_steps:
        filtered = precision[recall >= t]
        p_at_t = float(np.max(filtered) if len(filtered) > 0 else 0.0)
        ap += p_at_t / 11
    return ap

def compute_map(predictions: List[dict], ground_truths: List[dict], num_classes: int, iou_threshold: float = 0.5) -> Tuple[float, List[float]]:
    """
    For each class: accumulate TP/FP across dataset, compute PR curve, integrate AP.
    Returns: (mAP, [AP per class])
    """
    per_class_ap = []
    for c in range(num_classes):
        all_scores = []
        all_tp = []
        total_gt = 0

        for idx in range(len(predictions)):
            pred_mask = np.array(predictions[idx]['labels']) == c
            pred_boxes = np.array(predictions[idx]['boxes'])[pred_mask]
            pred_scores = np.array(predictions[idx]['scores'])[pred_mask]

            gt_mask = np.array(ground_truths[idx]['labels']) == c
            gt_boxes = np.array(ground_truths[idx]['boxes'])[gt_mask]
            total_gt += len(gt_boxes)

            if len(pred_boxes) == 0: continue
            if len(gt_boxes) == 0: 
                all_scores.extend(pred_scores.tolist())
                all_tp.extend([0] * len(pred_boxes))
                continue

            iou = compute_iou(torch.tensor(pred_boxes, dtype=torch.float32), torch.tensor(gt_boxes, dtype=torch.float32)).numpy()
            gt_matched = set()
            for n in range(len(pred_boxes)):
                best_gt = int(np.argmax(iou[n]))
                best_iou = iou[n, best_gt]
                all_scores.append(float(pred_scores[n]))
                if best_iou >= iou_threshold and best_gt not in gt_matched:
                    all_tp.append(1)
                    gt_matched.add(best_gt)
                else:
                    all_tp.append(0)

        if total_gt == 0 or len(all_scores) == 0:
            per_class_ap.append(0.0)
            continue
        
        order = np.argsort(-np.array(all_scores))
        tp_sorted = np.array(all_tp)[order]
        cumulative = np.cumsum(tp_sorted)
        precision = cumulative / (np.arange(len(cumulative)) + 1)
        recall = cumulative / total_gt
        per_class_ap.append(compute_ap(precision, recall))
    
    mAP = float(np.mean(per_class_ap) if per_class_ap else 0.0)
    
    return mAP, per_class_ap
    
