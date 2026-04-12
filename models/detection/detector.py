import torch
import torch.nn as nn
from models.backbone.resnet import ResNetBackbone
from models.detection.fpn import FPN
from models.detection.head import DetectionHead
from models.detection.anchors import AnchorGenerator
from models.detection.box_utils import decode_boxes, nms
from typing import Tuple, List


class FPNDetector(nn.Module):
    def __init__(
        self,
        backbone: ResNetBackbone,
        fpn: FPN,
        head: DetectionHead,
        anchor_generator: AnchorGenerator,
        num_classes: int,
        score_threshold: float = 0.05,
        nms_threshold: float = 0.5,
        max_detections: int = 100,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.fpn = fpn
        self.head = head
        self.anchor_generator = anchor_generator
        self.num_classes = num_classes
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections

    def forward(
        self, images: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Training mode: returns (cls_logits, bbox_deltas, anchors)
        Eval mode: calls postprocess() and returns (boxes, scores, labels) per image
        """
        # Backbone yields (C3, C4, C5) multi-scale features.
        features = self.backbone(images)
        # FPN returns (P5, P4, P3) per fpn.py's forward signature.
        fpn_features = self.fpn(features)
        cls_logits, bbox_deltas = self.head(list(fpn_features))

        feature_map_sizes = [(f.shape[-2], f.shape[-1]) for f in fpn_features]
        image_size = (images.shape[-2], images.shape[-1])
        anchors = self.anchor_generator.generate_all(feature_map_sizes, image_size).to(images.device)

        if self.training:
            return cls_logits, bbox_deltas, anchors
        return self.postprocess(cls_logits, bbox_deltas, anchors, image_size)

    def postprocess(
        self,
        cls_logits: List[torch.Tensor],
        bbox_deltas: List[torch.Tensor],
        anchors: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        1. Decode deltas → absolute boxes
        2. Sigmoid scores, threshold
        3. Apply NMS per class
        4. Return top-k detections
        """
        cls_logits_cat = torch.cat(cls_logits, dim=1)  # (B, N, C)
        bbox_deltas_cat = torch.cat(bbox_deltas, dim=1)  # (B, N, 4)
        B = cls_logits_cat.shape[0]
        device = cls_logits_cat.device
        img_h, img_w = image_size

        all_boxes: List[torch.Tensor] = []
        all_scores: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []

        for b in range(B):
            scores_b = torch.sigmoid(cls_logits_cat[b])  # (N, C)
            deltas_b = bbox_deltas_cat[b]  # (N, 4)
            boxes_b = decode_boxes(anchors, deltas_b).to(device)

            # Clip decoded boxes to image bounds.
            boxes_b[:, 0].clamp_(0, img_w)
            boxes_b[:, 1].clamp_(0, img_h)
            boxes_b[:, 2].clamp_(0, img_w)
            boxes_b[:, 3].clamp_(0, img_h)

            img_boxes: List[torch.Tensor] = []
            img_scores: List[torch.Tensor] = []
            img_labels: List[torch.Tensor] = []

            for c in range(self.num_classes):
                cls_scores = scores_b[:, c]
                keep_mask = cls_scores >= self.score_threshold
                if keep_mask.sum().item() == 0:
                    continue
                cand_boxes = boxes_b[keep_mask]
                cand_scores = cls_scores[keep_mask]

                # Guard against box_utils.nms crash on empty input (Bug #6).
                if cand_boxes.numel() == 0:
                    continue

                nms_keep = nms(cand_boxes, cand_scores, self.nms_threshold)
                img_boxes.append(cand_boxes[nms_keep])
                img_scores.append(cand_scores[nms_keep])
                img_labels.append(
                    torch.full((nms_keep.numel(),), c, dtype=torch.long, device=device)
                )

            if len(img_boxes) > 0:
                final_boxes = torch.cat(img_boxes, dim=0)
                final_scores = torch.cat(img_scores, dim=0)
                final_labels = torch.cat(img_labels, dim=0)

                # Keep only top-k detections overall (by score).
                if final_scores.numel() > self.max_detections:
                    topk = torch.topk(final_scores, self.max_detections)
                    final_boxes = final_boxes[topk.indices]
                    final_scores = topk.values
                    final_labels = final_labels[topk.indices]
            else:
                final_boxes = torch.zeros((0, 4), device=device)
                final_scores = torch.zeros((0,), device=device)
                final_labels = torch.zeros((0,), dtype=torch.long, device=device)

            all_boxes.append(final_boxes)
            all_scores.append(final_scores)
            all_labels.append(final_labels)

        return all_boxes, all_scores, all_labels
