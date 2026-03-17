"""
Unit tests for Phase 2 detection utilities: models/detection/box_utils.py.

Tests use synthetic tensors only — no nuScenes data required.
Safe to run in CI without GPU.

Coverage:
  - compute_iou    : shape, values, identical boxes, non-overlapping, partial overlap, single box
  - match_anchors_to_gt : shape/type of return, fg/bg assignment, empty gt_boxes (known crash)
  - encode_boxes   : round-trip with decode_boxes, delta magnitudes, identity anchor==gt
  - decode_boxes   : shape, value range (x2 > x1, y2 > y1), round-trip identity
  - nms            : keeps highest-scored box, removes overlapping, empty boxes (known crash),
                     single box, non-overlapping all kept, threshold boundary
"""

import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _boxes(*xyxy_tuples: tuple) -> torch.Tensor:
    """Convenience: build (N,4) float32 tensor from xyxy tuples."""
    return torch.tensor(list(xyxy_tuples), dtype=torch.float32)


# ---------------------------------------------------------------------------
# compute_iou
# ---------------------------------------------------------------------------

class TestComputeIou:

    def test_output_shape(self):
        """(N,4) × (M,4) → (N,M) matrix."""
        from models.detection.box_utils import compute_iou
        a = _boxes((0, 0, 10, 10), (5, 5, 15, 15))      # N=2
        b = _boxes((0, 0, 10, 10), (3, 3, 8, 8), (20, 20, 30, 30))  # M=3
        iou = compute_iou(a, b)
        assert iou.shape == (2, 3), f"Expected (2,3), got {iou.shape}"

    def test_dtype_is_float32(self):
        from models.detection.box_utils import compute_iou
        a = _boxes((0, 0, 4, 4))
        b = _boxes((0, 0, 4, 4))
        assert compute_iou(a, b).dtype == torch.float32

    def test_identical_boxes_iou_is_one(self):
        """IoU of a box with itself must be exactly 1.0."""
        from models.detection.box_utils import compute_iou
        box = _boxes((10, 20, 50, 80))
        iou = compute_iou(box, box)
        assert torch.allclose(iou, torch.tensor([[1.0]])), \
            f"Identical boxes IoU={iou.item():.4f}, expected 1.0"

    def test_non_overlapping_boxes_iou_is_zero(self):
        """Completely separate boxes must have IoU = 0."""
        from models.detection.box_utils import compute_iou
        a = _boxes((0, 0, 5, 5))
        b = _boxes((10, 10, 20, 20))
        iou = compute_iou(a, b)
        assert torch.allclose(iou, torch.tensor([[0.0]])), \
            f"Non-overlapping boxes IoU={iou.item():.4f}, expected 0.0"

    def test_partial_overlap_known_value(self):
        """Two 10×10 boxes sharing a 5×5 corner overlap.
        intersection=25, union=100+100-25=175, IoU=25/175≈0.1429."""
        from models.detection.box_utils import compute_iou
        a = _boxes((0, 0, 10, 10))
        b = _boxes((5, 5, 15, 15))
        iou = compute_iou(a, b)
        expected = 25.0 / 175.0
        assert abs(iou[0, 0].item() - expected) < 1e-4, \
            f"IoU={iou[0,0].item():.5f}, expected {expected:.5f}"

    def test_contained_box_iou_equals_area_ratio(self):
        """Small box fully inside large box: IoU = area_small / area_large."""
        from models.detection.box_utils import compute_iou
        large = _boxes((0, 0, 10, 10))   # area=100
        small = _boxes((2, 2, 7, 7))     # area=25
        iou = compute_iou(large, small)
        expected = 25.0 / 100.0          # intersection=25, union=100
        assert abs(iou[0, 0].item() - expected) < 1e-4

    def test_iou_values_in_range_zero_to_one(self):
        """All IoU values must be in [0, 1]."""
        from models.detection.box_utils import compute_iou
        torch.manual_seed(42)
        a = torch.rand(8, 4)
        # Ensure x2 > x1, y2 > y1
        a[:, 2] = a[:, 0] + a[:, 2].abs() + 0.01
        a[:, 3] = a[:, 1] + a[:, 3].abs() + 0.01
        b = torch.rand(5, 4)
        b[:, 2] = b[:, 0] + b[:, 2].abs() + 0.01
        b[:, 3] = b[:, 1] + b[:, 3].abs() + 0.01
        iou = compute_iou(a, b)
        assert (iou >= 0).all(), "IoU matrix contains negative values"
        assert (iou <= 1).all(), "IoU matrix contains values > 1"

    def test_iou_matrix_is_not_necessarily_symmetric(self):
        """compute_iou(a, b)[i,j] should equal compute_iou(b, a)[j,i]."""
        from models.detection.box_utils import compute_iou
        a = _boxes((0, 0, 6, 6), (2, 2, 8, 8))
        b = _boxes((1, 1, 5, 5))
        iou_ab = compute_iou(a, b)
        iou_ba = compute_iou(b, a)
        assert torch.allclose(iou_ab, iou_ba.T, atol=1e-5)

    def test_single_box_vs_single_box(self):
        """Edge case: N=1, M=1 → output (1,1)."""
        from models.detection.box_utils import compute_iou
        a = _boxes((0, 0, 4, 4))
        b = _boxes((2, 2, 6, 6))
        iou = compute_iou(a, b)
        assert iou.shape == (1, 1)
        # intersection = 2×2=4, union=16+16-4=28
        assert abs(iou[0, 0].item() - 4.0 / 28.0) < 1e-4

    def test_touching_boxes_zero_iou(self):
        """Boxes that share only an edge have zero intersection area."""
        from models.detection.box_utils import compute_iou
        a = _boxes((0, 0, 5, 5))
        b = _boxes((5, 0, 10, 5))   # shares x=5 edge
        iou = compute_iou(a, b)
        assert torch.allclose(iou, torch.tensor([[0.0]]))


# ---------------------------------------------------------------------------
# match_anchors_to_gt
# ---------------------------------------------------------------------------

class TestMatchAnchorsToGt:

    def test_output_lengths_match_num_anchors(self):
        """Returns two length-N sequences for N anchors."""
        from models.detection.box_utils import match_anchors_to_gt
        anchors = _boxes((0, 0, 10, 10), (5, 5, 15, 15), (20, 20, 30, 30))
        gt = _boxes((0, 0, 10, 10))
        indices, labels = match_anchors_to_gt(anchors, gt)
        assert len(indices) == 3
        assert len(labels) == 3

    def test_perfect_match_is_foreground(self):
        """Anchor identical to GT must get label=1 (fg)."""
        from models.detection.box_utils import match_anchors_to_gt
        anchors = _boxes((0, 0, 10, 10))
        gt = _boxes((0, 0, 10, 10))
        indices, labels = match_anchors_to_gt(anchors, gt, iou_threshold=0.5)
        assert labels[0] == 1

    def test_no_overlap_is_background(self):
        """Anchor with zero IoU vs all GT boxes must get label=0 (bg)."""
        from models.detection.box_utils import match_anchors_to_gt
        anchors = _boxes((100, 100, 110, 110))
        gt = _boxes((0, 0, 10, 10))
        indices, labels = match_anchors_to_gt(anchors, gt, iou_threshold=0.5)
        assert labels[0] == 0

    def test_threshold_boundary_below_is_background(self):
        """IoU just below threshold → background."""
        from models.detection.box_utils import match_anchors_to_gt
        # anchor (0,0,10,10), gt (6,0,16,10): intersection=4×10=40,
        # union=100+100-40=160, IoU=0.25 < 0.5
        anchors = _boxes((0, 0, 10, 10))
        gt = _boxes((6, 0, 16, 10))
        indices, labels = match_anchors_to_gt(anchors, gt, iou_threshold=0.5)
        assert labels[0] == 0

    def test_best_gt_index_is_correct(self):
        """Each anchor is matched to the GT with highest IoU."""
        from models.detection.box_utils import match_anchors_to_gt
        anchors = _boxes((0, 0, 10, 10))
        # gt[0] far away, gt[1] perfect match
        gt = _boxes((50, 50, 60, 60), (0, 0, 10, 10))
        indices, labels = match_anchors_to_gt(anchors, gt, iou_threshold=0.5)
        assert indices[0] == 1, \
            f"Expected best match idx=1 (perfect overlap), got {indices[0]}"

    def test_multiple_anchors_assigned_independently(self):
        """Each anchor gets its own best GT match."""
        from models.detection.box_utils import match_anchors_to_gt
        anchors = _boxes((0, 0, 10, 10), (20, 20, 30, 30))
        gt = _boxes((0, 0, 10, 10), (20, 20, 30, 30))
        indices, labels = match_anchors_to_gt(anchors, gt, iou_threshold=0.5)
        assert indices[0] == 0
        assert indices[1] == 1
        assert labels[0] == 1
        assert labels[1] == 1

    def test_empty_gt_boxes_returns_all_background(self):
        """Empty gt_boxes should return all-background labels without crashing."""
        from models.detection.box_utils import match_anchors_to_gt
        anchors = _boxes((0, 0, 10, 10), (5, 5, 15, 15))
        gt = torch.zeros((0, 4), dtype=torch.float32)
        indices, labels = match_anchors_to_gt(anchors, gt)
        assert len(labels) == 2
        assert all(l == 0 for l in labels), f"All labels should be bg (0), got {labels}"

    def test_return_type_annotation_vs_reality(self):
        """Documents bug #2: declared Tuple[Tensor, Tensor] but returns (list, list).
        This test will FAIL if the implementation is corrected to return tensors —
        at that point update the assertion to check isinstance(..., torch.Tensor)."""
        from models.detection.box_utils import match_anchors_to_gt
        anchors = _boxes((0, 0, 10, 10))
        gt = _boxes((0, 0, 10, 10))
        indices, labels = match_anchors_to_gt(anchors, gt)
        # Currently both are plain Python lists — documenting current behavior
        assert isinstance(indices, list), \
            "indices is now a Tensor — update this test and downstream callers"
        assert isinstance(labels, list), \
            "labels is now a Tensor — update this test and downstream callers"


# ---------------------------------------------------------------------------
# encode_boxes
# ---------------------------------------------------------------------------

class TestEncodeBoxes:

    def test_output_shape(self):
        """(N,4) anchors + (N,4) gt → (N,4) deltas."""
        from models.detection.box_utils import encode_boxes
        torch.manual_seed(42)
        anchors = _boxes((0, 0, 10, 10), (5, 5, 20, 20))
        gt = _boxes((1, 1, 11, 11), (5, 5, 20, 20))
        deltas = encode_boxes(anchors, gt)
        assert deltas.shape == (2, 4), f"Expected (2,4), got {deltas.shape}"

    def test_dtype_is_float32(self):
        from models.detection.box_utils import encode_boxes
        anchors = _boxes((0, 0, 10, 10))
        gt = _boxes((0, 0, 10, 10))
        deltas = encode_boxes(anchors, gt)
        assert deltas.dtype == torch.float32

    def test_identity_anchor_equals_gt_gives_zero_xy_deltas(self):
        """When anchor == GT, dx=0, dy=0, dw=0, dh=0."""
        from models.detection.box_utils import encode_boxes
        box = _boxes((10, 20, 50, 80))
        deltas = encode_boxes(box, box)
        assert torch.allclose(deltas, torch.zeros(1, 4), atol=1e-5), \
            f"Identity encode gave non-zero deltas: {deltas}"

    def test_dx_sign_correct(self):
        """GT center to the right of anchor → dx > 0."""
        from models.detection.box_utils import encode_boxes
        # anchor center=(5,5), gt center=(10,5) — shifted right
        anchor = _boxes((0, 0, 10, 10))
        gt = _boxes((5, 0, 15, 10))
        deltas = encode_boxes(anchor, gt)
        dx = deltas[0, 0].item()
        assert dx > 0, f"Expected dx > 0 for rightward shift, got dx={dx:.4f}"

    def test_dw_is_log_ratio(self):
        """dw = log(w_gt / w_anchor) — verify against manual calculation."""
        from models.detection.box_utils import encode_boxes
        import math
        anchor = _boxes((0, 0, 10, 10))   # w_a=10
        gt = _boxes((0, 0, 20, 10))       # w_g=20 → dw=log(2)
        deltas = encode_boxes(anchor, gt)
        dw = deltas[0, 2].item()
        assert abs(dw - math.log(2)) < 1e-4, \
            f"dw={dw:.5f}, expected log(2)={math.log(2):.5f}"

    def test_dh_is_log_ratio(self):
        """dh = log(h_gt / h_anchor)."""
        from models.detection.box_utils import encode_boxes
        import math
        anchor = _boxes((0, 0, 10, 10))   # h_a=10
        gt = _boxes((0, 0, 10, 40))       # h_g=40 → dh=log(4)
        deltas = encode_boxes(anchor, gt)
        dh = deltas[0, 3].item()
        assert abs(dh - math.log(4)) < 1e-4, \
            f"dh={dh:.5f}, expected log(4)={math.log(4):.5f}"

    def test_no_nan_or_inf_in_output(self):
        """Well-formed inputs must not produce NaN or Inf deltas."""
        from models.detection.box_utils import encode_boxes
        torch.manual_seed(42)
        anchors = _boxes((0, 0, 10, 10), (5, 5, 25, 25), (0, 0, 100, 50))
        gt = _boxes((1, 1, 12, 12), (8, 6, 30, 30), (10, 5, 90, 45))
        deltas = encode_boxes(anchors, gt)
        assert not torch.isnan(deltas).any(), "NaN in encode_boxes output"
        assert not torch.isinf(deltas).any(), "Inf in encode_boxes output"

    def test_single_anchor(self):
        """Edge case: N=1 must return (1,4)."""
        from models.detection.box_utils import encode_boxes
        anchor = _boxes((0, 0, 10, 10))
        gt = _boxes((2, 2, 8, 8))
        deltas = encode_boxes(anchor, gt)
        assert deltas.shape == (1, 4)


# ---------------------------------------------------------------------------
# decode_boxes
# ---------------------------------------------------------------------------

class TestDecodeBoxes:

    def test_output_shape(self):
        """(N,4) anchors + (N,4) deltas → (N,4) decoded boxes."""
        from models.detection.box_utils import decode_boxes
        anchors = _boxes((0, 0, 10, 10), (5, 5, 20, 20))
        deltas = torch.zeros(2, 4)
        boxes = decode_boxes(anchors, deltas)
        assert boxes.shape == (2, 4)

    def test_dtype_is_float32(self):
        from models.detection.box_utils import decode_boxes
        anchors = _boxes((0, 0, 10, 10))
        deltas = torch.zeros(1, 4)
        assert decode_boxes(anchors, deltas).dtype == torch.float32

    def test_zero_deltas_returns_anchors(self):
        """Decoding zero deltas must recover the original anchors exactly."""
        from models.detection.box_utils import decode_boxes
        anchors = _boxes((10, 20, 50, 80), (5, 5, 25, 35))
        deltas = torch.zeros(2, 4)
        boxes = decode_boxes(anchors, deltas)
        assert torch.allclose(boxes, anchors, atol=1e-4), \
            f"Zero deltas did not recover anchors.\nGot: {boxes}\nExpected: {anchors}"

    def test_round_trip_encode_decode(self):
        """encode then decode must recover the original GT boxes."""
        from models.detection.box_utils import encode_boxes, decode_boxes
        torch.manual_seed(42)
        anchors = _boxes((0, 0, 10, 10), (5, 5, 25, 30), (10, 10, 80, 60))
        gt = _boxes((1, 1, 12, 12), (8, 7, 28, 33), (15, 12, 75, 55))
        deltas = encode_boxes(anchors, gt)
        recovered = decode_boxes(anchors, deltas)
        assert torch.allclose(recovered, gt, atol=1e-4), \
            f"Round-trip failed.\nExpected: {gt}\nGot: {recovered}"

    def test_decoded_boxes_have_positive_width_height(self):
        """Decoded boxes must satisfy x2 > x1 and y2 > y1."""
        from models.detection.box_utils import decode_boxes
        torch.manual_seed(42)
        anchors = _boxes((0, 0, 10, 10), (5, 5, 20, 20), (3, 3, 15, 15))
        # Small positive/negative deltas
        deltas = torch.tensor([[0.1, -0.1, 0.2, -0.2],
                                [-0.3, 0.3, 0.0, 0.1],
                                [0.0, 0.0, 0.5, 0.5]], dtype=torch.float32)
        boxes = decode_boxes(anchors, deltas)
        assert (boxes[:, 2] > boxes[:, 0]).all(), "Decoded box has x2 <= x1"
        assert (boxes[:, 3] > boxes[:, 1]).all(), "Decoded box has y2 <= y1"

    def test_no_nan_or_inf_in_output(self):
        from models.detection.box_utils import decode_boxes
        anchors = _boxes((0, 0, 10, 10), (5, 5, 25, 25))
        deltas = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                                [-0.1, -0.2, 0.0, 0.0]], dtype=torch.float32)
        boxes = decode_boxes(anchors, deltas)
        assert not torch.isnan(boxes).any()
        assert not torch.isinf(boxes).any()

    def test_single_anchor(self):
        from models.detection.box_utils import decode_boxes
        anchor = _boxes((0, 0, 10, 10))
        delta = torch.zeros(1, 4)
        boxes = decode_boxes(anchor, delta)
        assert boxes.shape == (1, 4)


# ---------------------------------------------------------------------------
# nms
# ---------------------------------------------------------------------------

class TestNms:

    def test_single_box_is_kept(self):
        """One box in → one index out."""
        from models.detection.box_utils import nms
        boxes = _boxes((0, 0, 10, 10))
        scores = torch.tensor([0.9])
        kept = nms(boxes, scores)
        assert kept.shape == (1,)
        assert kept[0].item() == 0

    def test_non_overlapping_all_kept(self):
        """Boxes with zero mutual IoU — all should survive NMS."""
        from models.detection.box_utils import nms
        boxes = _boxes((0, 0, 5, 5), (10, 10, 15, 15), (20, 20, 25, 25))
        scores = torch.tensor([0.9, 0.8, 0.7])
        kept = nms(boxes, scores, iou_threshold=0.5)
        assert set(kept.tolist()) == {0, 1, 2}, \
            f"All non-overlapping boxes should be kept, got {kept.tolist()}"

    def test_identical_boxes_only_highest_score_kept(self):
        """Two identical boxes → keep only the one with higher score."""
        from models.detection.box_utils import nms
        boxes = _boxes((0, 0, 10, 10), (0, 0, 10, 10))
        scores = torch.tensor([0.6, 0.9])
        kept = nms(boxes, scores, iou_threshold=0.5)
        assert len(kept) == 1
        assert kept[0].item() == 1, \
            f"Expected highest-score box (idx=1) to be kept, got idx={kept[0].item()}"

    def test_highly_overlapping_lower_score_suppressed(self):
        """High IoU pair → only highest score survives."""
        from models.detection.box_utils import nms
        # box0 and box1 overlap heavily; box0 has higher score
        boxes = _boxes((0, 0, 10, 10), (1, 1, 11, 11), (50, 50, 60, 60))
        scores = torch.tensor([0.95, 0.7, 0.8])
        kept = nms(boxes, scores, iou_threshold=0.3)
        kept_list = kept.tolist()
        assert 0 in kept_list, "Highest-scored box 0 should be kept"
        assert 1 not in kept_list, "Overlapping lower-score box 1 should be suppressed"
        assert 2 in kept_list, "Non-overlapping box 2 should be kept"

    def test_result_sorted_by_score_descending(self):
        """Kept indices must appear in descending score order."""
        from models.detection.box_utils import nms
        boxes = _boxes((0, 0, 5, 5), (10, 10, 15, 15), (20, 20, 25, 25))
        scores = torch.tensor([0.5, 0.9, 0.7])
        kept = nms(boxes, scores, iou_threshold=0.5)
        # All non-overlapping; expected order: 1 (0.9), 2 (0.7), 0 (0.5)
        assert kept.tolist() == [1, 2, 0], \
            f"Expected score-descending order [1,2,0], got {kept.tolist()}"

    def test_output_is_1d_tensor(self):
        """Return type must be a 1D LongTensor (or IntTensor)."""
        from models.detection.box_utils import nms
        boxes = _boxes((0, 0, 5, 5), (10, 10, 15, 15))
        scores = torch.tensor([0.8, 0.6])
        kept = nms(boxes, scores)
        assert isinstance(kept, torch.Tensor)
        assert kept.dim() == 1

    def test_iou_threshold_controls_suppression(self):
        """Low threshold suppresses more aggressively than high threshold."""
        from models.detection.box_utils import nms
        # Two boxes with moderate overlap
        boxes = _boxes((0, 0, 10, 10), (3, 3, 13, 13))
        scores = torch.tensor([0.9, 0.8])
        kept_strict = nms(boxes, scores, iou_threshold=0.1)
        kept_loose = nms(boxes, scores, iou_threshold=0.9)
        assert len(kept_strict) == 1, "Low threshold should suppress the second box"
        assert len(kept_loose) == 2, "High threshold should keep both boxes"

    def test_empty_boxes_returns_empty_tensor(self):
        """Empty input should return an empty long tensor without crashing."""
        from models.detection.box_utils import nms
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        scores = torch.zeros((0,), dtype=torch.float32)
        kept = nms(boxes, scores)
        assert isinstance(kept, torch.Tensor)
        assert len(kept) == 0
        assert kept.dtype == torch.long

    def test_all_below_threshold_only_top_kept(self):
        """With threshold=0.0, every box suppresses every other → only top-1 kept."""
        from models.detection.box_utils import nms
        # Slightly overlapping boxes — any overlap will trigger suppression at threshold=0
        boxes = _boxes((0, 0, 10, 10), (1, 1, 9, 9), (2, 2, 8, 8))
        scores = torch.tensor([0.9, 0.5, 0.3])
        kept = nms(boxes, scores, iou_threshold=0.0)
        assert len(kept) == 1
        assert kept[0].item() == 0

    def test_no_nan_in_scores_does_not_crash(self):
        """Verify numeric stability with very small score differences."""
        from models.detection.box_utils import nms
        boxes = _boxes((0, 0, 10, 10), (0, 0, 10, 10))
        scores = torch.tensor([1e-7, 2e-7])
        kept = nms(boxes, scores, iou_threshold=0.5)
        assert len(kept) == 1
