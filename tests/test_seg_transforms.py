"""
Unit tests for the segmentation transforms in data/transforms.py.

Verify:
  - get_seg_train/val_transforms accept image= and mask= kwargs
  - image output is (3, H, W) float32, mask output is (H, W)
  - the mask is resampled with NEAREST — class IDs are never blended into
    new (non-existent) values
  - the val pipeline is deterministic

No nuScenes data or GPU required.
"""

import numpy as np
import pytest
import torch

INPUT_H, INPUT_W = 448, 800
NUM_CLASSES = 5


def _random_image(h=900, w=1600):
    rng = np.random.RandomState(42)
    return (rng.rand(h, w, 3) * 255).astype(np.uint8)


def _random_mask(h=900, w=1600):
    rng = np.random.RandomState(7)
    return rng.randint(0, NUM_CLASSES, (h, w), dtype=np.uint8)


class TestSegValTransforms:

    @pytest.fixture
    def transform(self):
        from data.transforms import get_seg_val_transforms
        return get_seg_val_transforms(input_h=INPUT_H, input_w=INPUT_W)

    def test_image_output_shape_and_dtype(self, transform):
        out = transform(image=_random_image(), mask=_random_mask())
        assert out["image"].shape == (3, INPUT_H, INPUT_W)
        assert out["image"].dtype == torch.float32

    def test_mask_output_shape(self, transform):
        out = transform(image=_random_image(), mask=_random_mask())
        assert out["mask"].shape == (INPUT_H, INPUT_W)

    def test_mask_class_ids_not_blended(self, transform):
        """A mask with only IDs {0, 4} must resample to a subset of {0, 4}."""
        img = _random_image()
        mask = np.where(_random_mask() > 2, 4, 0).astype(np.uint8)
        out = transform(image=img, mask=mask)
        vals = set(np.unique(out["mask"].numpy()).tolist())
        assert vals.issubset({0, 4}), f"NEAREST broken — got {vals}"

    def test_deterministic(self, transform):
        img, mask = _random_image(), _random_mask()
        a = transform(image=img.copy(), mask=mask.copy())
        b = transform(image=img.copy(), mask=mask.copy())
        assert torch.allclose(a["image"], b["image"])
        assert torch.equal(a["mask"], b["mask"])


class TestSegTrainTransforms:

    @pytest.fixture
    def transform(self):
        from data.transforms import get_seg_train_transforms
        return get_seg_train_transforms(input_h=INPUT_H, input_w=INPUT_W)

    def test_image_output_shape_and_dtype(self, transform):
        out = transform(image=_random_image(), mask=_random_mask())
        assert out["image"].shape == (3, INPUT_H, INPUT_W)
        assert out["image"].dtype == torch.float32

    def test_mask_output_shape(self, transform):
        out = transform(image=_random_image(), mask=_random_mask())
        assert out["mask"].shape == (INPUT_H, INPUT_W)

    def test_mask_stays_valid_class_ids_under_augmentation(self, transform):
        """Across augmentation seeds, the mask must contain only valid IDs.

        Augmentation may introduce 0 (border fill) but never a value >= NUM_CLASSES.
        """
        for seed in range(5):
            rng = np.random.RandomState(seed)
            img = (rng.rand(900, 1600, 3) * 255).astype(np.uint8)
            mask = rng.randint(0, NUM_CLASSES, (900, 1600), dtype=np.uint8)
            out = transform(image=img, mask=mask)
            vals = np.unique(out["mask"].numpy())
            assert vals.min() >= 0 and vals.max() < NUM_CLASSES, \
                f"seed={seed}: mask IDs out of range {vals}"

    def test_image_and_mask_spatial_agreement(self, transform):
        """Image and mask must share the same spatial size after transforms."""
        out = transform(image=_random_image(), mask=_random_mask())
        assert out["image"].shape[-2:] == out["mask"].shape[-2:]