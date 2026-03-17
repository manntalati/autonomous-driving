import torch
from typing import Tuple
import numpy as np

def compute_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise IoU between two sets of boxes.
    Args: boxes_a — (N, 4) in [x1,y1,x2,y2]; boxes_b — (M, 4) in [x1,y1,x2,y2].
    Returns: (N, M) float32 tensor where entry [i,j] = IoU(boxes_a[i], boxes_b[j]).
    """
    matrix = np.zeros((len(boxes_a), len(boxes_b)))
    for i in range(len(boxes_a)):
        for j in range(len(boxes_b)):
            x1_inter = max(boxes_a[i][0], boxes_b[j][0])
            y1_inter = max(boxes_a[i][1], boxes_b[j][1])
            x2_inter = min(boxes_a[i][2], boxes_b[j][2])
            y2_inter = min(boxes_a[i][3], boxes_b[j][3])
            inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
            area_a = (boxes_a[i][2] - boxes_a[i][0]) * (boxes_a[i][3] - boxes_a[i][1])
            area_b = (boxes_b[j][2] - boxes_b[j][0]) * (boxes_b[j][3] - boxes_b[j][1])
            union_area = area_a + area_b - inter_area
            matrix[i][j] = inter_area / union_area if union_area > 0 else 0.0

    return torch.tensor(matrix, dtype=torch.float32)

def match_anchors_to_gt(anchors: torch.Tensor, gt_boxes: torch.Tensor, iou_threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    For each anchor, find the best matching GT box by IoU and assign fg/bg label.
    Args: anchors — (N, 4) anchor boxes; gt_boxes — (M, 4) ground truth boxes;
          iou_threshold — IoU >= threshold → label 1 (fg), else label 0 (bg).
    Returns: (matched_gt_indices, match_labels) — both length-N lists;
             matched_gt_indices[i] = index into gt_boxes of best match for anchor i.
    """
    if len(gt_boxes) == 0:
        return ([], [0] * len(anchors))
    matrix = compute_iou(anchors, gt_boxes)
    matched_gt_indices = []
    match_labels = []
    for i in range(len(matrix)):
        best_match = torch.argmax(matrix[i]).item()
        best_iou = matrix[i][best_match]
        matched_gt_indices.append(best_match)
        if best_iou >= iou_threshold: match_labels.append(1)
        else: match_labels.append(0)

    return (matched_gt_indices, match_labels)

def encode_boxes(anchors: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert absolute GT boxes into (dx, dy, dw, dh) deltas relative to anchors.
    Both inputs are in [x1,y1,x2,y2]; internally converted to center format for encoding.
    Args: anchors — (N, 4); gt_boxes — (N, 4) matched GT box for each anchor.
    Returns: (N, 4) float32 tensor of deltas [dx, dy, dw, dh] — what the network learns to predict.
    """
    deltas = []
    for i in range(len(anchors)):
        cx_a, cy_a = (anchors[i][0] + anchors[i][2]) / 2, (anchors[i][1] + anchors[i][3]) / 2
        cx_g, cy_g = (gt_boxes[i][0] + gt_boxes[i][2]) / 2, (gt_boxes[i][1] + gt_boxes[i][3]) / 2
        w_a, h_a = anchors[i][2] - anchors[i][0], anchors[i][3] - anchors[i][1]
        w_g, h_g = gt_boxes[i][2] - gt_boxes[i][0], gt_boxes[i][3] - gt_boxes[i][1]
        dx = (cx_g - cx_a) / w_a
        dy = (cy_g - cy_a) / h_a
        dw = torch.log(w_g / w_a)
        dh = torch.log(h_g / h_a)
        deltas.append((dx,dy,dw,dh))
    return torch.stack([torch.stack([dx, dy, dw, dh]) for dx, dy, dw, dh in deltas])

def decode_boxes(anchors: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """
    Inverse of encode_boxes. Apply predicted deltas to anchors to get final boxes.
    Args: anchors — (N, 4) in [x1,y1,x2,y2]; deltas — (N, 4) predicted [dx,dy,dw,dh].
    Returns: (N, 4) float32 tensor of decoded boxes in [x1,y1,x2,y2].
    """
    boxes = []
    for i in range(len(anchors)):
        dx,dy,dw,dh = deltas[i]
        cx_a, cy_a = (anchors[i][0] + anchors[i][2]) / 2, (anchors[i][1] + anchors[i][3]) / 2
        w_a, h_a = anchors[i][2] - anchors[i][0], anchors[i][3] - anchors[i][1]
        cx_g, cy_g = dx * w_a + cx_a, dy * h_a + cy_a
        w_g, h_g = w_a * torch.exp(dw), h_a * torch.exp(dh)
        x1,y1,x2,y2 = cx_g - w_g / 2, cy_g - h_g / 2, cx_g + w_g / 2, cy_g + h_g / 2
        boxes.append((x1,y1,x2,y2))
    return torch.stack([torch.stack([x1, y1, x2, y2]) for x1, y1, x2, y2 in boxes])

def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5) -> torch.Tensor:
    """
    Non-maximum suppression. Iteratively keep the highest-scored box and remove overlapping duplicates.
    Args: boxes — (N, 4) in [x1,y1,x2,y2]; scores — (N,) confidence scores;
          iou_threshold — boxes with IoU >= threshold vs kept box are suppressed.
    Returns: 1D tensor of indices of kept boxes (sorted by score descending).
    """
    order = torch.argsort(scores, descending=True)
    if len(order) == 0:
        return torch.tensor([], dtype=torch.long)
    kept = []
    while len(order) > 0:
        i = order[0]
        kept.append(i)
        iou = compute_iou(boxes[i].unsqueeze(0), boxes[order[1:]])[0]
        order = order[1:][iou < iou_threshold]
    return torch.stack(kept)
