"""
Unit tests for Phase 2 detection components:
  - models/detection/anchors.py  — AnchorGenerator
  - models/detection/fpn.py      — FPN
  - models/detection/head.py     — DetectionHead

All tests use synthetic tensors only — no nuScenes data required.
Safe to run in CI without GPU.
"""

import math
import pytest
import torch
import torch.nn as nn


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_backbone_features(batch: int = 2) -> tuple:
    """Return synthetic (C3, C4, C5) feature maps at typical FPN input sizes.

    Using project resolution 448x800 with ResNet strides 8/16/32:
      C3: stride 8  → 56x100 spatial, 128 channels
      C4: stride 16 → 28x50  spatial, 256 channels
      C5: stride 32 → 14x25  spatial, 512 channels
    """
    torch.manual_seed(42)
    c3 = torch.randn(batch, 128, 56, 100)
    c4 = torch.randn(batch, 256, 28, 50)
    c5 = torch.randn(batch, 512, 14, 25)
    return c3, c4, c5


# ──────────────────────────────────────────────────────────────────────────────
# AnchorGenerator — generate_for_level
# ──────────────────────────────────────────────────────────────────────────────

class TestAnchorGeneratorForLevel:

    @pytest.fixture
    def gen(self):
        from models.detection.anchors import AnchorGenerator
        return AnchorGenerator(
            scales=[128.0, 256.0, 512.0],
            aspect_ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32],
        )

    def test_output_shape_single_cell(self, gen):
        """1x1 feature map with 3 aspect ratios → (1*1*3, 4) = (3, 4)."""
        out = gen.generate_for_level(feature_h=1, feature_w=1, stride=8, scale=128.0)
        assert out.shape == (3, 4), f"Expected (3, 4), got {out.shape}"

    def test_output_shape_grid(self, gen):
        """HxW feature map with 3 ratios → (H*W*3, 4)."""
        H, W = 4, 6
        out = gen.generate_for_level(feature_h=H, feature_w=W, stride=16, scale=256.0)
        assert out.shape == (H * W * 3, 4), f"Expected ({H*W*3}, 4), got {out.shape}"

    def test_output_is_float32(self, gen):
        """Anchors must be float32 to match other box tensors in the codebase.

        NOTE: The current implementation returns float64 (default torch.tensor
        dtype from a list of Python floats). This test documents the expected
        behavior — it will FAIL until the implementation is fixed with
        `.float()` or `dtype=torch.float32` in the tensor call.
        """
        out = gen.generate_for_level(feature_h=2, feature_w=2, stride=8, scale=64.0)
        assert out.dtype == torch.float32, (
            f"generate_for_level returned {out.dtype}, expected float32. "
            "Fix: add dtype=torch.float32 to torch.tensor() call in anchors.py."
        )

    def test_anchor_columns_are_x1_y1_x2_y2(self, gen):
        """x1 < x2 and y1 < y2 for all anchors (positive width and height)."""
        out = gen.generate_for_level(feature_h=3, feature_w=4, stride=8, scale=64.0).float()
        assert (out[:, 2] > out[:, 0]).all(), "x2 <= x1 for some anchors"
        assert (out[:, 3] > out[:, 1]).all(), "y2 <= y1 for some anchors"

    def test_center_of_first_anchor_is_half_stride(self, gen):
        """First anchor's center must be at (0.5*stride, 0.5*stride)."""
        stride = 8
        out = gen.generate_for_level(feature_h=2, feature_w=2, stride=stride, scale=64.0).float()
        # Center of first anchor: cx = (col+0.5)*stride = 0.5*8 = 4
        #                         cy = (row+0.5)*stride = 0.5*8 = 4
        cx = (out[0, 0] + out[0, 2]) / 2.0
        cy = (out[0, 1] + out[0, 3]) / 2.0
        assert abs(cx.item() - 0.5 * stride) < 1e-3, f"cx={cx:.3f}, expected {0.5*stride}"
        assert abs(cy.item() - 0.5 * stride) < 1e-3, f"cy={cy:.3f}, expected {0.5*stride}"

    def test_anchor_width_matches_scale_and_ratio(self, gen):
        """For ratio=1.0 and scale=S: w = S*sqrt(1) = S, h = S/sqrt(1) = S."""
        scale = 64.0
        # aspect_ratios=[0.5, 1.0, 2.0], so ratio=1.0 is index 1
        out = gen.generate_for_level(feature_h=1, feature_w=1, stride=8, scale=scale).float()
        # anchor for ratio=1.0 is the second anchor (index 1)
        anchor_ratio1 = out[1]
        w = (anchor_ratio1[2] - anchor_ratio1[0]).item()
        h = (anchor_ratio1[3] - anchor_ratio1[1]).item()
        assert abs(w - scale) < 1e-3, f"width={w:.3f}, expected {scale}"
        assert abs(h - scale) < 1e-3, f"height={h:.3f}, expected {scale}"

    def test_anchor_width_height_for_ratio_2(self, gen):
        """For ratio=2.0: w = S*sqrt(2), h = S/sqrt(2)."""
        scale = 64.0
        out = gen.generate_for_level(feature_h=1, feature_w=1, stride=8, scale=scale).float()
        # ratio=2.0 is index 2
        anchor = out[2]
        w = (anchor[2] - anchor[0]).item()
        h = (anchor[3] - anchor[1]).item()
        assert abs(w - scale * math.sqrt(2.0)) < 1e-3
        assert abs(h - scale / math.sqrt(2.0)) < 1e-3

    def test_number_of_anchors_is_h_times_w_times_num_ratios(self, gen):
        """Total anchors = H * W * len(aspect_ratios)."""
        H, W = 7, 5
        num_ratios = len(gen.aspect_ratios)  # 3
        out = gen.generate_for_level(feature_h=H, feature_w=W, stride=32, scale=512.0)
        assert out.shape[0] == H * W * num_ratios

    def test_second_cell_center_offset_by_one_stride(self, gen):
        """The anchor in cell (row=0, col=1) for the first ratio must have
        center at (1.5*stride, 0.5*stride).

        Loop order in generate_for_level is: ratio → row → col.
        With feature_h=2, feature_w=3 and 3 ratios:
          index 0: ratio=0.5, row=0, col=0  → cx=0.5s, cy=0.5s
          index 1: ratio=0.5, row=0, col=1  → cx=1.5s, cy=0.5s   ← this one
          index 2: ratio=0.5, row=0, col=2  → cx=2.5s, cy=0.5s
          index 3: ratio=0.5, row=1, col=0  → cx=0.5s, cy=1.5s
          ...
        """
        stride = 16
        out = gen.generate_for_level(feature_h=2, feature_w=3, stride=stride, scale=128.0).float()
        # Index 1: ratio=0.5, row=0, col=1 → cx=1.5*stride, cy=0.5*stride
        anchor = out[1]
        cx = (anchor[0] + anchor[2]) / 2.0
        cy = (anchor[1] + anchor[3]) / 2.0
        assert abs(cx.item() - 1.5 * stride) < 1e-3, \
            f"cx={cx:.3f}, expected {1.5*stride}"
        assert abs(cy.item() - 0.5 * stride) < 1e-3, \
            f"cy={cy:.3f}, expected {0.5*stride}"

    def test_no_nan_in_output(self, gen):
        out = gen.generate_for_level(feature_h=4, feature_w=4, stride=8, scale=64.0).float()
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


# ──────────────────────────────────────────────────────────────────────────────
# AnchorGenerator — generate_all
# ──────────────────────────────────────────────────────────────────────────────

class TestAnchorGeneratorAll:

    @pytest.fixture
    def gen(self):
        from models.detection.anchors import AnchorGenerator
        return AnchorGenerator(
            scales=[128.0, 256.0, 512.0],
            aspect_ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32],
        )

    def test_output_shape_three_levels(self, gen):
        """Total anchors = sum(H_i * W_i * num_ratios) across all levels."""
        feature_map_sizes = [(56, 100), (28, 50), (14, 25)]
        image_size = (448, 800)
        num_ratios = len(gen.aspect_ratios)  # 3
        expected = sum(h * w * num_ratios for h, w in feature_map_sizes)
        out = gen.generate_all(feature_map_sizes, image_size)
        assert out.shape == (expected, 4), f"Expected ({expected}, 4), got {out.shape}"

    def test_output_has_four_columns(self, gen):
        """Each anchor is represented by [x1, y1, x2, y2]."""
        out = gen.generate_all([(4, 4), (2, 2), (1, 1)], (32, 32))
        assert out.shape[1] == 4

    def test_anchors_clamped_to_image_bounds(self, gen):
        """All anchor coordinates must lie within [0, img_w] and [0, img_h]."""
        image_size = (448, 800)
        out = gen.generate_all([(56, 100), (28, 50), (14, 25)], image_size).float()
        img_h, img_w = image_size
        assert (out[:, 0] >= 0).all() and (out[:, 0] <= img_w).all(), "x1 out of bounds"
        assert (out[:, 1] >= 0).all() and (out[:, 1] <= img_h).all(), "y1 out of bounds"
        assert (out[:, 2] >= 0).all() and (out[:, 2] <= img_w).all(), "x2 out of bounds"
        assert (out[:, 3] >= 0).all() and (out[:, 3] <= img_h).all(), "y2 out of bounds"

    def test_single_level_matches_generate_for_level(self, gen):
        """generate_all with one level must equal generate_for_level output (after clamp)."""
        H, W, stride, scale = 4, 6, 8, 128.0
        image_size = (64, 80)
        from models.detection.anchors import AnchorGenerator
        single_gen = AnchorGenerator(
            scales=[scale], aspect_ratios=gen.aspect_ratios, strides=[stride]
        )
        all_anchors = single_gen.generate_all([(H, W)], image_size).float()
        level_anchors = single_gen.generate_for_level(H, W, stride, scale).float()
        # Clamp level anchors to match generate_all behavior
        img_h, img_w = image_size
        level_anchors[:, 0].clamp_(0, img_w)
        level_anchors[:, 1].clamp_(0, img_h)
        level_anchors[:, 2].clamp_(0, img_w)
        level_anchors[:, 3].clamp_(0, img_h)
        assert torch.allclose(all_anchors, level_anchors, atol=1e-4)

    def test_output_is_2d_tensor(self, gen):
        out = gen.generate_all([(2, 2), (1, 1)], (16, 16))
        assert out.dim() == 2

    def test_no_nan_in_output(self, gen):
        out = gen.generate_all([(4, 8), (2, 4), (1, 2)], (32, 64)).float()
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_concatenates_all_levels(self, gen):
        """Output length must be strictly greater than any single level's count."""
        feature_map_sizes = [(4, 4), (2, 2), (1, 1)]
        out = gen.generate_all(feature_map_sizes, (32, 32))
        from models.detection.anchors import AnchorGenerator
        single_gen = AnchorGenerator([128.0], gen.aspect_ratios, [8])
        max_single = max(
            single_gen.generate_for_level(h, w, 8, 128.0).shape[0]
            for h, w in feature_map_sizes
        )
        assert out.shape[0] > max_single, (
            "generate_all output is not larger than the biggest single level — "
            "concatenation may be broken"
        )


# ──────────────────────────────────────────────────────────────────────────────
# FPN — forward pass
# ──────────────────────────────────────────────────────────────────────────────

class TestFPN:

    @pytest.fixture
    def fpn(self):
        from models.detection.fpn import FPN
        return FPN(in_channels=[128, 256, 512], out_channels=256)

    @pytest.fixture
    def features(self):
        return _make_backbone_features(batch=2)

    # ── Module structure ────────────────────────────────────────────────────

    def test_lateral_convs_count(self, fpn):
        """One lateral conv per input level (3 total)."""
        assert len(fpn.lateral_convs) == 3

    def test_output_convs_count(self, fpn):
        """One output conv per level (3 total)."""
        assert len(fpn.output_convs) == 3

    def test_lateral_convs_are_1x1(self, fpn):
        """Lateral convs must use kernel_size=1."""
        for conv in fpn.lateral_convs:
            assert isinstance(conv, nn.Conv2d)
            assert conv.kernel_size == (1, 1), \
                f"Lateral conv kernel is {conv.kernel_size}, expected (1, 1)"

    def test_output_convs_are_3x3(self, fpn):
        """Output convs must use kernel_size=3."""
        for conv in fpn.output_convs:
            assert isinstance(conv, nn.Conv2d)
            assert conv.kernel_size == (3, 3), \
                f"Output conv kernel is {conv.kernel_size}, expected (3, 3)"

    def test_lateral_convs_reduce_to_out_channels(self, fpn):
        """Each lateral conv must map its input channels to out_channels=256."""
        in_channels = [128, 256, 512]
        for conv, in_ch in zip(fpn.lateral_convs, in_channels):
            assert conv.in_channels == in_ch
            assert conv.out_channels == 256

    # ── Output shape ────────────────────────────────────────────────────────

    def test_forward_returns_three_tensors(self, fpn, features):
        """FPN forward must return a 3-tuple."""
        out = fpn(features)
        assert len(out) == 3

    def test_p5_channel_dim_is_out_channels(self, fpn, features):
        """P5 (coarsest level) must have out_channels=256."""
        p5, p4, p3 = fpn(features)
        assert p5.shape[1] == 256, f"P5 channels={p5.shape[1]}, expected 256"

    def test_p4_channel_dim_is_out_channels(self, fpn, features):
        p5, p4, p3 = fpn(features)
        assert p4.shape[1] == 256

    def test_p3_channel_dim_is_out_channels(self, fpn, features):
        p5, p4, p3 = fpn(features)
        assert p3.shape[1] == 256

    def test_p5_spatial_matches_c5(self, fpn, features):
        """P5 spatial dimensions must match C5 input (no upsampling at the top)."""
        c3, c4, c5 = features
        p5, p4, p3 = fpn(features)
        assert p5.shape[2:] == c5.shape[2:], \
            f"P5 spatial {p5.shape[2:]} != C5 spatial {c5.shape[2:]}"

    def test_p4_spatial_matches_c4(self, fpn, features):
        """P4 spatial dimensions must match C4 (top-down upsample preserves C4 size)."""
        c3, c4, c5 = features
        p5, p4, p3 = fpn(features)
        assert p4.shape[2:] == c4.shape[2:], \
            f"P4 spatial {p4.shape[2:]} != C4 spatial {c4.shape[2:]}"

    def test_p3_spatial_matches_c3(self, fpn, features):
        """P3 spatial dimensions must match C3."""
        c3, c4, c5 = features
        p5, p4, p3 = fpn(features)
        assert p3.shape[2:] == c3.shape[2:], \
            f"P3 spatial {p3.shape[2:]} != C3 spatial {c3.shape[2:]}"

    def test_batch_dim_preserved(self, fpn, features):
        """Batch dimension must pass through unchanged."""
        B = features[0].shape[0]
        for feat in fpn(features):
            assert feat.shape[0] == B

    # ── Gradient flow ────────────────────────────────────────────────────────

    def test_gradients_flow_through_all_levels(self, fpn, features):
        """Gradients from P3 + P4 + P5 loss must reach all FPN parameters."""
        p5, p4, p3 = fpn(features)
        loss = p5.sum() + p4.sum() + p3.sum()
        loss.backward()
        for name, param in fpn.named_parameters():
            assert param.grad is not None, f"No gradient for FPN parameter: {name}"

    def test_gradients_flow_from_p3_alone(self, fpn, features):
        """Backprop through P3 alone reaches the parameters involved in computing P3.

        The top-down chain is: lateral5 → p5 → upsample → add with lateral4 → p4
        → upsample → add with lateral3 → p3 → output_convs[0].

        Therefore backpropping through P3 reaches:
          - output_convs[0]  (direct head of P3)
          - lateral_convs[0] (C3 lateral)
          - lateral_convs[1] (C4 lateral, via top-down add)
          - lateral_convs[2] (C5 lateral, via top-down add chain)
          - output_convs[1]  (NOT reached — p4_ret is a separate output)
          - output_convs[2]  (NOT reached — p5_ret is a separate output)

        This test verifies the lateral conv chain is intact and that output_convs[0]
        receives a gradient.
        """
        _p5, _p4, p3 = fpn(features)
        p3.sum().backward()
        # These must have gradients (part of P3's computation path)
        must_have_grad = [
            'lateral_convs.0.weight', 'lateral_convs.0.bias',
            'lateral_convs.1.weight', 'lateral_convs.1.bias',
            'lateral_convs.2.weight', 'lateral_convs.2.bias',
            'output_convs.0.weight',  'output_convs.0.bias',
        ]
        for name in must_have_grad:
            param = dict(fpn.named_parameters())[name]
            assert param.grad is not None, \
                f"Expected gradient for {name} when backpropping through P3, but got None"

    # ── Top-down pathway sanity ──────────────────────────────────────────────

    def test_no_nan_or_inf_in_output(self, fpn, features):
        for feat in fpn(features):
            assert not torch.isnan(feat).any(), "NaN in FPN output"
            assert not torch.isinf(feat).any(), "Inf in FPN output"

    def test_output_changes_when_input_changes(self, fpn):
        """FPN output must differ when the input features change."""
        torch.manual_seed(0)
        c3a = torch.randn(1, 128, 14, 25)
        c4a = torch.randn(1, 256, 7, 13)
        c5a = torch.randn(1, 512, 4, 7)
        torch.manual_seed(1)
        c3b = torch.randn(1, 128, 14, 25)
        c4b = torch.randn(1, 256, 7, 13)
        c5b = torch.randn(1, 512, 4, 7)
        fpn.eval()
        with torch.no_grad():
            out_a = fpn((c3a, c4a, c5a))
            out_b = fpn((c3b, c4b, c5b))
        for a, b in zip(out_a, out_b):
            assert not torch.allclose(a, b, atol=1e-6), \
                "FPN output is identical for different inputs — something is wrong"

    def test_custom_out_channels(self):
        """out_channels parameter is respected for all lateral and output convs."""
        from models.detection.fpn import FPN
        fpn = FPN(in_channels=[64, 128, 256], out_channels=128)
        torch.manual_seed(42)
        c3 = torch.randn(1, 64, 8, 8)
        c4 = torch.randn(1, 128, 4, 4)
        c5 = torch.randn(1, 256, 2, 2)
        p5, p4, p3 = fpn((c3, c4, c5))
        assert p5.shape[1] == 128
        assert p4.shape[1] == 128
        assert p3.shape[1] == 128

    def test_train_eval_both_run_without_error(self, fpn, features):
        """FPN has no BN/Dropout but both modes must run cleanly."""
        fpn.train()
        fpn(features)
        fpn.eval()
        with torch.no_grad():
            fpn(features)


# ──────────────────────────────────────────────────────────────────────────────
# DetectionHead — forward pass
# ──────────────────────────────────────────────────────────────────────────────

class TestDetectionHead:

    # Shared parameters across tests
    IN_CHANNELS = 256
    NUM_ANCHORS = 3
    NUM_CLASSES = 3  # car / pedestrian / cyclist

    @pytest.fixture
    def head(self):
        from models.detection.head import DetectionHead
        return DetectionHead(
            in_channels=self.IN_CHANNELS,
            num_anchors=self.NUM_ANCHORS,
            num_classes=self.NUM_CLASSES,
            num_convs=4,
        )

    @pytest.fixture
    def single_level_input(self):
        """Single FPN level: (B=2, C=256, H=14, W=25)."""
        torch.manual_seed(42)
        return [torch.randn(2, self.IN_CHANNELS, 14, 25)]

    @pytest.fixture
    def three_level_input(self):
        """Three FPN levels mimicking project FPN output."""
        torch.manual_seed(42)
        return [
            torch.randn(2, self.IN_CHANNELS, 14, 25),   # P5
            torch.randn(2, self.IN_CHANNELS, 28, 50),   # P4
            torch.randn(2, self.IN_CHANNELS, 56, 100),  # P3
        ]

    # ── Module structure ────────────────────────────────────────────────────

    def test_tower_has_num_convs_conv_layers(self, head):
        """Tower must contain exactly num_convs Conv2d layers."""
        conv_layers = [m for m in head.tower.modules() if isinstance(m, nn.Conv2d)]
        assert len(conv_layers) == 4, \
            f"Expected 4 Conv2d in tower, found {len(conv_layers)}"

    def test_tower_has_relu_after_each_conv(self, head):
        """Tower must contain num_convs ReLU activations."""
        relus = [m for m in head.tower.modules() if isinstance(m, nn.ReLU)]
        assert len(relus) == 4

    def test_cls_head_output_channels(self, head):
        """cls_head must produce num_anchors * num_classes channels."""
        assert head.cls_head.out_channels == self.NUM_ANCHORS * self.NUM_CLASSES

    def test_reg_head_output_channels(self, head):
        """reg_head must produce num_anchors * 4 channels."""
        assert head.reg_head.out_channels == self.NUM_ANCHORS * 4

    # ── Return structure ────────────────────────────────────────────────────

    def test_forward_returns_two_lists(self, head, single_level_input):
        """forward must return a 2-tuple of lists: (cls_logits_list, bbox_deltas_list)."""
        cls_list, reg_list = head(single_level_input)
        assert isinstance(cls_list, list)
        assert isinstance(reg_list, list)

    def test_list_length_matches_num_levels(self, head, three_level_input):
        """Output lists must have one entry per input FPN level."""
        cls_list, reg_list = head(three_level_input)
        assert len(cls_list) == 3
        assert len(reg_list) == 3

    # ── Output shapes ────────────────────────────────────────────────────────

    def test_cls_logits_shape_single_level(self, head, single_level_input):
        """cls_logits shape must be (B, H*W*A, num_classes)."""
        B, _, H, W = single_level_input[0].shape
        A, C = self.NUM_ANCHORS, self.NUM_CLASSES
        cls_list, _ = head(single_level_input)
        expected = (B, H * W * A, C)
        assert cls_list[0].shape == expected, \
            f"cls_logits shape {cls_list[0].shape}, expected {expected}"

    def test_bbox_deltas_shape_single_level(self, head, single_level_input):
        """bbox_deltas shape must be (B, H*W*A, 4)."""
        B, _, H, W = single_level_input[0].shape
        A = self.NUM_ANCHORS
        _, reg_list = head(single_level_input)
        expected = (B, H * W * A, 4)
        assert reg_list[0].shape == expected, \
            f"bbox_deltas shape {reg_list[0].shape}, expected {expected}"

    def test_cls_logits_shape_three_levels(self, head, three_level_input):
        """All three cls_logit tensors have correct shapes."""
        cls_list, _ = head(three_level_input)
        A, C = self.NUM_ANCHORS, self.NUM_CLASSES
        for i, feat in enumerate(three_level_input):
            B, _, H, W = feat.shape
            expected = (B, H * W * A, C)
            assert cls_list[i].shape == expected, \
                f"Level {i}: cls shape {cls_list[i].shape}, expected {expected}"

    def test_bbox_deltas_shape_three_levels(self, head, three_level_input):
        """All three bbox_delta tensors have shape (B, H*W*A, 4)."""
        _, reg_list = head(three_level_input)
        A = self.NUM_ANCHORS
        for i, feat in enumerate(three_level_input):
            B, _, H, W = feat.shape
            expected = (B, H * W * A, 4)
            assert reg_list[i].shape == expected, \
                f"Level {i}: reg shape {reg_list[i].shape}, expected {expected}"

    def test_last_dim_is_four_for_reg(self, head, three_level_input):
        """Regression output last dimension must always be 4 (dx, dy, dw, dh)."""
        _, reg_list = head(three_level_input)
        for i, reg in enumerate(reg_list):
            assert reg.shape[-1] == 4, \
                f"Level {i}: last dim of bbox_deltas is {reg.shape[-1]}, expected 4"

    def test_last_dim_is_num_classes_for_cls(self, head, three_level_input):
        """Classification output last dimension must always be num_classes."""
        cls_list, _ = head(three_level_input)
        for i, cls in enumerate(cls_list):
            assert cls.shape[-1] == self.NUM_CLASSES, \
                f"Level {i}: last dim of cls_logits is {cls.shape[-1]}, expected {self.NUM_CLASSES}"

    def test_batch_size_1_runs_without_error(self, head):
        """Single-item batch must not crash."""
        feat = [torch.randn(1, self.IN_CHANNELS, 7, 7)]
        cls_list, reg_list = head(feat)
        assert cls_list[0].shape[0] == 1
        assert reg_list[0].shape[0] == 1

    # ── Weight sharing across levels ────────────────────────────────────────

    def test_weights_are_shared_across_levels(self, head, three_level_input):
        """DetectionHead uses the same tower/cls_head/reg_head for every FPN level —
        parameter count must be independent of the number of levels.
        Verify by checking that tower parameters are the same objects regardless
        of which level runs first."""
        # Run a forward pass and check that tower weights haven't been duplicated
        tower_param_ids_before = {id(p) for p in head.tower.parameters()}
        head(three_level_input)
        tower_param_ids_after = {id(p) for p in head.tower.parameters()}
        assert tower_param_ids_before == tower_param_ids_after, \
            "Tower parameter object IDs changed — parameters may have been duplicated"

    # ── Gradient flow ────────────────────────────────────────────────────────

    def test_gradients_flow_to_tower_and_heads(self, head, three_level_input):
        """Loss summed over all levels must push gradients to every named parameter."""
        cls_list, reg_list = head(three_level_input)
        loss = sum(c.sum() for c in cls_list) + sum(r.sum() for r in reg_list)
        loss.backward()
        for name, param in head.named_parameters():
            assert param.grad is not None, f"No gradient for: {name}"

    # ── Numeric sanity ───────────────────────────────────────────────────────

    def test_no_nan_or_inf_in_cls_output(self, head, three_level_input):
        cls_list, _ = head(three_level_input)
        for i, cls in enumerate(cls_list):
            assert not torch.isnan(cls).any(), f"NaN in cls_logits at level {i}"
            assert not torch.isinf(cls).any(), f"Inf in cls_logits at level {i}"

    def test_no_nan_or_inf_in_reg_output(self, head, three_level_input):
        _, reg_list = head(three_level_input)
        for i, reg in enumerate(reg_list):
            assert not torch.isnan(reg).any(), f"NaN in bbox_deltas at level {i}"
            assert not torch.isinf(reg).any(), f"Inf in bbox_deltas at level {i}"

    def test_train_eval_mode_both_run(self, head, three_level_input):
        head.train()
        head(three_level_input)
        head.eval()
        with torch.no_grad():
            head(three_level_input)

    def test_different_num_convs(self):
        """num_convs parameter controls tower depth — 2 convs should work too."""
        from models.detection.head import DetectionHead
        head2 = DetectionHead(in_channels=128, num_anchors=2, num_classes=3, num_convs=2)
        feat = [torch.randn(1, 128, 4, 4)]
        cls_list, reg_list = head2(feat)
        assert cls_list[0].shape == (1, 4 * 4 * 2, 3)
        assert reg_list[0].shape == (1, 4 * 4 * 2, 4)
