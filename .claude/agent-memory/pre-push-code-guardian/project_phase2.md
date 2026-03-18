---
name: project_phase2
description: Phase 2 detection utilities status — box_utils.py bugs found and documented, tests written
type: project
---

Phase 2 detection utility status as of 2026-03-16:

## Completed components (stubs only — student to implement)
- models/detection/anchors.py — AnchorGenerator skeleton
- models/detection/fpn.py — FPN skeleton
- models/detection/head.py — DetectionHead skeleton
- models/detection/losses.py — focal_loss, smooth_l1_loss, DetectionLoss skeletons
- models/detection/detector.py — FPNDetector skeleton
- models/detection/map.py — compute_ap, compute_map skeletons
- models/detection/train_detector.py — training script skeleton

## Reviewed and tested: models/detection/box_utils.py

All 5 functions are implemented (not stubs): compute_iou, match_anchors_to_gt, encode_boxes, decode_boxes, nms.
43 unit tests added in tests/test_box_utils.py — all pass as of 2026-03-16.

## Known Bugs in box_utils.py (reported, not fixed)

- CRITICAL #3: match_anchors_to_gt crashes with RuntimeError when gt_boxes is empty (shape 0,4).
  torch.argmax on a zero-element tensor raises. Common real-world case (frames with no annotations).
  Test: test_empty_gt_boxes_crashes — currently passes because it expects the crash.

- CRITICAL #6: nms crashes with RuntimeError when boxes input is empty (shape 0,4).
  torch.stack([]) on empty list raises. Must be fixed before nms is called in inference on empty frames.
  Test: test_empty_boxes_crashes — currently passes because it expects the crash.

- WARNING #1: compute_iou uses Python loops + NumPy internally. Will crash on CUDA tensors
  (numpy cannot accept CUDA tensors). Must be replaced with vectorized torch ops before GPU training.

- WARNING #2: match_anchors_to_gt declares return type Tuple[Tensor, Tensor] but actually returns
  (list, list). Type hint is wrong. Downstream code using tensor operations on the return will silently
  work but is fragile.
  Test: test_return_type_annotation_vs_reality — documents current list behavior.

- WARNING #4: encode_boxes has no guard against zero-area anchors (w_a=0 or h_a=0). Division by zero
  produces inf/nan deltas that will silently corrupt training loss.

- WARNING #5: encode_boxes and decode_boxes use element-wise Python loops — thousands of
  device→host syncs per forward pass on GPU. Not a correctness bug but a severe performance issue.

- WARNING #7: nms suppresses boxes with IoU exactly == iou_threshold (uses strict `<`). This differs
  from torchvision NMS which uses strict `>` (keeps ties). Behavior is internally consistent but
  undocumented and may surprise users expecting torchvision semantics.

## Test conventions established for Phase 2

- _boxes() helper builds (N,4) float32 tensors from xyxy tuples — use in future detection tests
- Known-crash bugs are tested with pytest.raises(Exception) and include docstring noting when to flip assertion
- Round-trip tests (encode -> decode) are the primary correctness check for box coding
- All tests use synthetic boxes, no nuScenes data, CI-safe

**Why:** box_utils.py was student-implemented; full review + test coverage requested before Phase 2 push.

**How to apply:** The two Critical bugs will cause crashes in any real inference loop with empty frames.
Do not let student push detector code without first fixing bugs #3 and #6. Remind student of
compute_iou CUDA incompatibility before GPU training begins.
