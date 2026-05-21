"""
Unit tests for models/segmentation/unet.py — U-Net segmentation head.

Tests use random tensors and a randomly-initialised ResNetBackbone (no ImageNet
download). Verify:
  - UpBlock output shape with and without a skip connection
  - UNet forward produces (B, num_classes, H, W) logits at input resolution
  - gradients flow end-to-end
  - UNet rejects a backbone that still has a classifier head

No nuScenes data or GPU required.
"""

import pytest
import torch

from models.backbone.resnet import ResNetBackbone
from models.segmentation.unet import UNet, UpBlock


class TestUpBlock:

    def test_with_skip_matches_skip_resolution(self):
        """With a skip, output spatial size == skip's, channels == out_channels."""
        block = UpBlock(in_channels=64, skip_channels=32, out_channels=16)
        x = torch.randn(2, 64, 8, 12)
        skip = torch.randn(2, 32, 16, 24)
        out = block(x, skip)
        assert out.shape == (2, 16, 16, 24)

    def test_without_skip_doubles_resolution(self):
        """Without a skip, output is a 2× upsample of x."""
        block = UpBlock(in_channels=64, skip_channels=0, out_channels=16)
        x = torch.randn(2, 64, 8, 12)
        out = block(x, None)
        assert out.shape == (2, 16, 16, 24)

    def test_skip_concat_channel_arithmetic(self):
        """conv1 must consume in_channels + skip_channels (concat) without error."""
        block = UpBlock(in_channels=128, skip_channels=64, out_channels=32)
        x = torch.randn(1, 128, 4, 4)
        skip = torch.randn(1, 64, 8, 8)
        out = block(x, skip)
        assert out.shape == (1, 32, 8, 8)


class TestUNet:

    @pytest.fixture
    def model(self):
        return UNet(ResNetBackbone(), num_classes=5)

    def test_forward_output_shape(self, model):
        """Output logits must be (B, num_classes, H, W) at input resolution."""
        x = torch.randn(2, 3, 64, 96)
        out = model(x)
        assert out.shape == (2, 5, 64, 96)

    def test_forward_matches_input_resolution(self, model):
        """Output H, W must equal the input H, W exactly (final upsample)."""
        x = torch.randn(1, 3, 128, 128)
        out = model(x)
        assert out.shape[-2:] == x.shape[-2:]

    def test_backward_flows(self, model):
        """Gradients must propagate to backbone and decoder parameters."""
        x = torch.randn(1, 3, 64, 64, requires_grad=True)
        out = model(x)
        out.sum().backward()
        assert x.grad is not None
        assert all(p.grad is not None for p in model.classifier.parameters())

    def test_num_classes_attribute(self, model):
        """num_classes attribute must match the classifier output channels."""
        assert model.num_classes == 5
        assert model.classifier.out_channels == 5

    def test_rejects_backbone_with_classifier_head(self):
        """A backbone built with num_classes set is not a feature extractor."""
        clf_backbone = ResNetBackbone(num_classes=10)
        with pytest.raises(AssertionError):
            UNet(clf_backbone, num_classes=5)

    def test_custom_decoder_channels(self):
        """decoder_channels argument must be respected."""
        model = UNet(ResNetBackbone(), num_classes=3, decoder_channels=(128, 64, 32))
        out = model(torch.randn(1, 3, 64, 64))
        assert out.shape == (1, 3, 64, 64)

    def test_eval_mode_is_deterministic(self, model):
        """In eval mode, repeated forwards on the same input must match."""
        model.eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            a = model(x)
            b = model(x)
        assert torch.allclose(a, b)