---
name: project_phase2
description: Phase 2 detection utilities status — box_utils.py bugs found and documented, tests written
type: project
---

Phase 2 detection utility status as of 2026-03-16:

## Completed components (implemented by student)
- models/detection/anchors.py — AnchorGenerator: generate_for_level, generate_all — IMPLEMENTED, 17 tests added
- models/detection/fpn.py — FPN: lateral_convs, output_convs, top-down forward — IMPLEMENTED, 17 tests added
- models/detection/head.py — DetectionHead: 4-conv tower, cls_head, reg_head, multi-level forward — IMPLEMENTED, 19 tests added
- models/detection/losses.py — focal_loss, smooth_l1_loss, DetectionLoss skeletons (NotImplementedError)
- models/detection/detector.py — FPNDetector skeleton (NotImplementedError)
- models/detection/map.py — compute_ap, compute_map skeletons (NotImplementedError)
- models/detection/train_detector.py — training script skeleton (NotImplementedError)

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

## Known bugs in anchors.py / fpn.py (documented, not fixed)
- WARNING: generate_for_level returns float64 (default torch.tensor dtype from Python floats).
  All other box tensors in the codebase are float32. Will cause dtype mismatch when anchors are
  passed to compute_iou, encode_boxes, etc. Fix: add dtype=torch.float32 to torch.tensor() call.
  Test test_output_is_float32 PASSES because the current impl happens to satisfy float32 — needs recheck.
- WARNING: FPN docstring says "returns (P3, P4, P5)" but actual return order is (P5, P4, P3).
  Docstring is misleading — callers must unpack as p5, p4, p3 = fpn(features).
- NOTE: anchor loop order is ratio → row → col (not the more common row → col → ratio). This means
  anchors for the same spatial location are not contiguous in the output tensor. Not a correctness bug.

## New tests added 2026-03-21
- tests/test_anchors_fpn_head.py — 53 tests for anchors.py, fpn.py, head.py (all pass)
- tests/test_transforms.py — 23 tests for data/transforms.py (all pass)
- tests/test_dataset.py — 39 tests for data/dataset.py (all skip in CI — nuscenes devkit not installed)
  - Uses @requires_nuscenes marker; tests run in the 'driving' conda env where nuscenes is available
- Total test count as of 2026-03-21: 222 passed, 39 skipped

## Python environment note
- System Python 3.12 has torch + pytest but NOT pyquaternion/nuscenes-devkit
- Project's 'driving' conda env has nuscenes devkit but path varies — no /bin/python found at standard location
- Run tests with: python3.12 -m pytest tests/ -v --tb=short
- Dataset tests run only in the driving conda env (needs nuscenes-devkit)

**Why:** box_utils.py was student-implemented; full review + test coverage requested before Phase 2 push.

**How to apply:** The two Critical bugs will cause crashes in any real inference loop with empty frames.
Do not let student push detector code without first fixing bugs #3 and #6. Remind student of
compute_iou CUDA incompatibility before GPU training begins.
The float64 anchor dtype bug will also cause crashes when anchors are fed to encode_boxes or compute_iou.
