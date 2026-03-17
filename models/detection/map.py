import numpy as np
from typing import List, Tuple

def compute_ap(precision: np.ndarray, recall: np.ndarray) -> float:
    """
    Area under the precision-recall curve (11-point interpolation or COCO-style).
    """
    pass

def compute_map(predictions: List[dict], ground_truths: List[dict], num_classes: int, iou_threshold: float = 0.5) -> Tuple[float, List[float]]:
    """
    For each class: accumulate TP/FP across dataset, compute PR curve, integrate AP.
    Returns: (mAP, [AP per class])
    """
    pass
