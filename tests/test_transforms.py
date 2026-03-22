"""
Unit tests for data/transforms.py — albumentations pipelines.

Tests use synthetic numpy images and dummy bounding boxes to verify:
  - output tensor shapes
  - pixel value normalization (ImageNet stats)
  - bbox pass-through for val pipeline
  - bbox handling edge cases (empty list, all boxes)
  - val pipeline determinism

No nuScenes data or GPU required.
"""

import numpy as np
import pytest
import torch

INPUT_H = 448
INPUT_W = 800

# ImageNet normalization constants (must match transforms.py)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _random_image(h: int = 900, w: int = 1600) -> np.ndarray:
    """Return a random uint8 RGB image of shape (H, W, 3)."""
    rng = np.random.RandomState(42)
    return (rng.rand(h, w, 3) * 255).astype(np.uint8)


def _valid_boxes_pascal_voc(img_h: int, img_w: int, n: int = 3):
    """Return n non-trivial bounding boxes in pascal_voc format [x1, y1, x2, y2].
    Boxes are well within image bounds so they survive min_visibility filtering.
    """
    step_x = img_w // (n + 1)
    step_y = img_h // (n + 1)
    boxes = []
    for i in range(1, n + 1):
        x1 = step_x * i - step_x // 2
        y1 = step_y * i - step_y // 2
        x2 = x1 + step_x // 2
        y2 = y1 + step_y // 2
        boxes.append([float(x1), float(y1), float(x2), float(y2)])
    return boxes


# ──────────────────────────────────────────────────────────────────────────────
# get_val_transforms
# ──────────────────────────────────────────────────────────────────────────────

class TestValTransforms:

    @pytest.fixture
    def transform(self):
        from data.transforms import get_val_transforms
        return get_val_transforms(input_h=INPUT_H, input_w=INPUT_W)

    def test_output_image_is_tensor(self, transform):
        """ToTensorV2 must convert numpy → torch.Tensor."""
        img = _random_image(900, 1600)
        result = transform(image=img, bboxes=[], labels=[])
        assert isinstance(result['image'], torch.Tensor)

    def test_output_image_shape_is_chw(self, transform):
        """Output image must have shape (3, INPUT_H, INPUT_W)."""
        img = _random_image(900, 1600)
        result = transform(image=img, bboxes=[], labels=[])
        assert result['image'].shape == (3, INPUT_H, INPUT_W), \
            f"Expected (3, {INPUT_H}, {INPUT_W}), got {result['image'].shape}"

    def test_output_dtype_is_float32(self, transform):
        """ToTensorV2 + Normalize produces float32."""
        img = _random_image(900, 1600)
        result = transform(image=img, bboxes=[], labels=[])
        assert result['image'].dtype == torch.float32

    def test_pixel_values_normalized_with_imagenet_stats(self, transform):
        """After normalization, pixels should be close to zero-mean unit-std."""
        # Use a solid-color image where we can compute expected output exactly
        # Pixel = 128 (uint8) → float32 = 128/255 ≈ 0.502
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = transform(image=img, bboxes=[], labels=[])
        out = result['image']   # (3, H, W)
        # Expected per-channel: (128/255 - mean[c]) / std[c]
        for c in range(3):
            pixel_val = 128.0 / 255.0
            expected = (pixel_val - MEAN[c]) / STD[c]
            actual_mean = out[c].mean().item()
            assert abs(actual_mean - expected) < 0.02, \
                f"Channel {c}: mean={actual_mean:.4f}, expected~{expected:.4f}"

    def test_boxes_survive_val_transform(self, transform):
        """Boxes that are well within the image must survive the val pipeline."""
        img = _random_image(900, 1600)
        boxes = _valid_boxes_pascal_voc(900, 1600, n=3)
        labels = [0, 1, 2]
        result = transform(image=img, bboxes=boxes, labels=labels)
        assert len(result['bboxes']) == 3, \
            f"Expected 3 boxes to survive, got {len(result['bboxes'])}"
        assert len(result['labels']) == 3

    def test_empty_boxes_do_not_crash(self, transform):
        """Empty bounding box list must not raise."""
        img = _random_image(900, 1600)
        result = transform(image=img, bboxes=[], labels=[])
        assert len(result['bboxes']) == 0
        assert len(result['labels']) == 0

    def test_boxes_rescaled_after_resize(self, transform):
        """After resize, box coordinates must be within [0, INPUT_W] x [0, INPUT_H]."""
        img = _random_image(900, 1600)
        boxes = _valid_boxes_pascal_voc(900, 1600, n=2)
        labels = [0, 1]
        result = transform(image=img, bboxes=boxes, labels=labels)
        for box in result['bboxes']:
            x1, y1, x2, y2 = box
            assert 0 <= x1 < x2 <= INPUT_W, f"Box x coords out of range: {x1}, {x2}"
            assert 0 <= y1 < y2 <= INPUT_H, f"Box y coords out of range: {y1}, {y2}"

    def test_val_transform_is_deterministic(self, transform):
        """Same input must produce identical output on repeated calls (no randomness)."""
        img = _random_image(300, 500)
        boxes = _valid_boxes_pascal_voc(300, 500, n=2)
        labels = [0, 1]
        r1 = transform(image=img.copy(), bboxes=list(boxes), labels=list(labels))
        r2 = transform(image=img.copy(), bboxes=list(boxes), labels=list(labels))
        assert torch.allclose(r1['image'], r2['image']), \
            "Val transform is not deterministic — augmentation may be leaking in"
        assert r1['bboxes'] == r2['bboxes']

    def test_custom_resolution(self):
        """get_val_transforms respects custom input_h / input_w arguments."""
        from data.transforms import get_val_transforms
        transform = get_val_transforms(input_h=224, input_w=224)
        img = _random_image(100, 100)
        result = transform(image=img, bboxes=[], labels=[])
        assert result['image'].shape == (3, 224, 224)

    def test_single_pixel_image_does_not_crash(self, transform):
        """Degenerate 1x1 image must not raise."""
        img = np.full((1, 1, 3), 100, dtype=np.uint8)
        result = transform(image=img, bboxes=[], labels=[])
        assert result['image'].shape == (3, INPUT_H, INPUT_W)

    def test_labels_preserved_with_boxes(self, transform):
        """Labels list must remain aligned with boxes list after transform."""
        img = _random_image(900, 1600)
        boxes = _valid_boxes_pascal_voc(900, 1600, n=3)
        labels = [0, 1, 2]
        result = transform(image=img, bboxes=boxes, labels=labels)
        # Verify labels and boxes are the same length
        assert len(result['bboxes']) == len(result['labels'])


# ──────────────────────────────────────────────────────────────────────────────
# get_train_transforms
# ──────────────────────────────────────────────────────────────────────────────

class TestTrainTransforms:

    @pytest.fixture
    def transform(self):
        from data.transforms import get_train_transforms
        return get_train_transforms(input_h=INPUT_H, input_w=INPUT_W)

    def test_output_image_is_tensor(self, transform):
        img = _random_image(900, 1600)
        result = transform(image=img, bboxes=[], labels=[])
        assert isinstance(result['image'], torch.Tensor)

    def test_output_image_shape_is_chw(self, transform):
        """Output must be (3, INPUT_H, INPUT_W) regardless of augmentation."""
        img = _random_image(900, 1600)
        result = transform(image=img, bboxes=[], labels=[])
        assert result['image'].shape == (3, INPUT_H, INPUT_W)

    def test_output_dtype_is_float32(self, transform):
        img = _random_image(900, 1600)
        result = transform(image=img, bboxes=[], labels=[])
        assert result['image'].dtype == torch.float32

    def test_empty_boxes_do_not_crash(self, transform):
        """Train pipeline must handle images with zero annotations."""
        img = _random_image(900, 1600)
        result = transform(image=img, bboxes=[], labels=[])
        assert len(result['bboxes']) == 0

    def test_boxes_stay_within_image_after_augmentation(self, transform):
        """Augmented boxes must still lie within the output image bounds."""
        rng = np.random.RandomState(0)
        # Run multiple times to cover different random augmentation branches
        for seed in range(5):
            img = (rng.rand(900, 1600, 3) * 255).astype(np.uint8)
            boxes = _valid_boxes_pascal_voc(900, 1600, n=4)
            labels = [0, 1, 2, 0]
            result = transform(image=img, bboxes=boxes, labels=labels)
            for box in result['bboxes']:
                x1, y1, x2, y2 = box
                assert x1 >= 0 and x2 <= INPUT_W, \
                    f"Box x out of range: [{x1}, {x2}] (seed={seed})"
                assert y1 >= 0 and y2 <= INPUT_H, \
                    f"Box y out of range: [{y1}, {y2}] (seed={seed})"

    def test_labels_and_boxes_aligned(self, transform):
        """After augmentation, labels and boxes must have the same count."""
        img = _random_image(900, 1600)
        boxes = _valid_boxes_pascal_voc(900, 1600, n=4)
        labels = [0, 1, 2, 0]
        result = transform(image=img, bboxes=boxes, labels=labels)
        assert len(result['bboxes']) == len(result['labels'])

    def test_has_bbox_params_configured(self, transform):
        """The train pipeline must have bbox_params so boxes are tracked."""
        # Verify by passing boxes and getting them back (not an error)
        img = _random_image(900, 1600)
        boxes = _valid_boxes_pascal_voc(900, 1600, n=1)
        result = transform(image=img, bboxes=boxes, labels=[0])
        # If bbox_params were missing, albumentations would raise a ValueError
        assert 'bboxes' in result

    def test_pixel_values_are_normalized(self, transform):
        """Pixel values must not be in [0, 255] after normalization."""
        img = _random_image(900, 1600)
        result = transform(image=img, bboxes=[], labels=[])
        out = result['image']
        # After ImageNet normalization, values are in roughly [-2.5, 2.5]
        # They must not still be in [0, 255] range
        assert out.max().item() < 10.0, \
            "Pixel max too high — normalization may not have been applied"
        assert out.min().item() > -10.0, \
            "Pixel min too low — something unexpected in normalization"

    def test_train_and_val_produce_same_spatial_shape(self):
        """Both pipelines must produce the same (3, H, W) output shape."""
        from data.transforms import get_train_transforms, get_val_transforms
        t_train = get_train_transforms(input_h=INPUT_H, input_w=INPUT_W)
        t_val = get_val_transforms(input_h=INPUT_H, input_w=INPUT_W)
        img = _random_image(900, 1600)
        r_train = t_train(image=img.copy(), bboxes=[], labels=[])
        r_val = t_val(image=img.copy(), bboxes=[], labels=[])
        assert r_train['image'].shape == r_val['image'].shape


# ──────────────────────────────────────────────────────────────────────────────
# Normalization constants sanity check
# ──────────────────────────────────────────────────────────────────────────────

class TestNormalizationConstants:

    def test_mean_values_match_imagenet(self):
        """transforms.py must use the standard ImageNet mean."""
        from data.transforms import MEAN
        expected = (0.485, 0.456, 0.406)
        for actual, exp in zip(MEAN, expected):
            assert abs(actual - exp) < 1e-4, \
                f"MEAN mismatch: {MEAN} vs {expected}"

    def test_std_values_match_imagenet(self):
        """transforms.py must use the standard ImageNet std."""
        from data.transforms import STD
        expected = (0.229, 0.224, 0.225)
        for actual, exp in zip(STD, expected):
            assert abs(actual - exp) < 1e-4, \
                f"STD mismatch: {STD} vs {expected}"

    def test_default_resolution_constants(self):
        """INPUT_H and INPUT_W must match the project specification (448x800)."""
        from data.transforms import INPUT_H, INPUT_W
        assert INPUT_H == 448, f"INPUT_H={INPUT_H}, expected 448"
        assert INPUT_W == 800, f"INPUT_W={INPUT_W}, expected 800"
