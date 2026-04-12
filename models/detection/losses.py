import torch
import torch.nn as nn
import torchvision
from typing import List, Tuple
from models.detection.box_utils import match_anchors_to_gt, encode_boxes


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """
    Focal loss — down-weights easy negatives to handle class imbalance.
    The gamma term is the key: (1 - p_t)^gamma shrinks gradient for confident predictions.
    Parameters:
        - logits: (B, N, num_classes) raw output from the model
        - targets: (B, N, num_classes) one-hot encoded GT labels
        - alpha: weight for the rare class (default 0.25)
        - gamma: focusing parameter to reduce loss for well-classified examples (default 2.0)
    Returns:
        - loss: per-element tensor of focal losses (same shape as logits)
    """
    return torchvision.ops.sigmoid_focal_loss(logits, targets, alpha, gamma, reduction="none")

def smooth_l1_loss(pred: torch.Tensor, target: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """
    Huber-style loss for box regression. Linear for large errors, quadratic for small.
    Robust to outlier boxes.
    Returns per-element tensor (same shape as pred).
    """
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    return loss


class DetectionLoss(nn.Module):
    def __init__(self, num_classes: int, cls_weight: float = 1.0, reg_weight: float = 1.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight

    def forward(
        self,
        cls_logits: List[torch.Tensor],
        bbox_deltas: List[torch.Tensor],
        anchors: torch.Tensor,
        gt_boxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, dict]:
        """
        1. Match anchors to GT (via box_utils.match_anchors_to_gt)
        2. Encode GT boxes to deltas (via box_utils.encode_boxes)
        3. Compute focal_loss on cls, smooth_l1 on reg (positives only)
        Returns total loss + dict of sub-losses for logging.
        """
        # Concatenate all FPN levels into (B, N_total, C) / (B, N_total, 4)
        cls_logits_cat = torch.cat(cls_logits, dim=1)
        bbox_deltas_cat = torch.cat(bbox_deltas, dim=1)

        B, N, _ = cls_logits_cat.shape
        device = cls_logits_cat.device

        if anchors.device != device:
            anchors = anchors.to(device)

        total_cls_loss = cls_logits_cat.new_zeros(())
        total_reg_loss = cls_logits_cat.new_zeros(())
        total_num_pos = 0

        for b in range(B):
            gt_b = gt_boxes[b].to(device) if gt_boxes[b].numel() > 0 else gt_boxes[b]
            labels_b = gt_labels[b].to(device) if gt_labels[b].numel() > 0 else gt_labels[b]

            matched_gt_idx, match_labels = match_anchors_to_gt(anchors, gt_b)
            # match_anchors_to_gt returns python lists; convert to tensors on correct device.
            match_labels_t = torch.as_tensor(match_labels, dtype=torch.long, device=device)
            if len(matched_gt_idx) > 0:
                matched_gt_idx_t = torch.as_tensor(matched_gt_idx, dtype=torch.long, device=device)
            else:
                matched_gt_idx_t = torch.zeros(N, dtype=torch.long, device=device)

            pos_mask = match_labels_t == 1
            num_pos = int(pos_mask.sum().item())
            total_num_pos += num_pos

            # Classification target: one-hot over positives, zeros elsewhere (sigmoid focal loss).
            cls_target = torch.zeros((N, self.num_classes), device=device, dtype=cls_logits_cat.dtype)
            if num_pos > 0 and labels_b.numel() > 0:
                pos_gt_idx = matched_gt_idx_t[pos_mask]
                pos_labels = labels_b[pos_gt_idx].long()
                pos_anchor_idx = torch.nonzero(pos_mask, as_tuple=False).squeeze(1)
                cls_target[pos_anchor_idx, pos_labels] = 1.0

            cls_loss_b = focal_loss(cls_logits_cat[b], cls_target).sum()
            total_cls_loss = total_cls_loss + cls_loss_b

            # Regression loss: positives only.
            if num_pos > 0 and gt_b.numel() > 0:
                pos_anchors = anchors[pos_mask]
                matched_boxes = gt_b[matched_gt_idx_t[pos_mask]]
                reg_target = encode_boxes(pos_anchors, matched_boxes).to(device)
                reg_pred = bbox_deltas_cat[b][pos_mask]
                reg_loss_b = smooth_l1_loss(reg_pred, reg_target).sum()
                total_reg_loss = total_reg_loss + reg_loss_b

        # Normalize by number of positives (standard RetinaNet convention).
        denom = max(total_num_pos, 1)
        cls_loss = total_cls_loss / denom
        reg_loss = total_reg_loss / denom
        total = self.cls_weight * cls_loss + self.reg_weight * reg_loss

        return total, {
            "loss": float(total.detach().item()),
            "cls_loss": float(cls_loss.detach().item()),
            "reg_loss": float(reg_loss.detach().item()) if isinstance(reg_loss, torch.Tensor) else float(reg_loss),
            "num_pos": total_num_pos,
        }
