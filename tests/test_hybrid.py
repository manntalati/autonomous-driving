"""
Unit tests for models/backbone/hybrid.py — hybrid CNN-ViT backbone.

Verify:
  - HybridCNNViT returns (C3, C4, C5) with the same channels/strides as
    ResNetBackbone (128/256/512 at strides 8/16/32) — the drop-in contract
  - it plugs into the Phase 3 UNet unchanged
  - embed_dim != 512 is rejected (UNet's decoder hardcodes a 512-channel C5)

Random tensors only — no nuScenes data, GPU, or weight downloads.
"""

import pytest
import torch

from models.backbone.hybrid import HybridCNNViT
from models.segmentation.unet import UNet


class TestHybridCNNViT:

    def test_returns_three_feature_maps(self):
        h = HybridCNNViT(image_size=(64, 64))
        out = h(torch.randn(2, 3, 64, 64))
        assert len(out) == 3

    def test_feature_map_channels_and_strides(self):
        """C3/C4/C5 must be 128/256/512 ch at strides 8/16/32 — ResNetBackbone contract."""
        h = HybridCNNViT(image_size=(64, 64))
        c3, c4, c5 = h(torch.randn(2, 3, 64, 64))
        assert c3.shape == (2, 128, 8, 8)    # stride 8
        assert c4.shape == (2, 256, 4, 4)    # stride 16
        assert c5.shape == (2, 512, 2, 2)    # stride 32

    def test_backbone_num_classes_is_none(self):
        """UNet asserts the backbone has num_classes is None."""
        assert HybridCNNViT().num_classes is None

    def test_rejects_non_512_embed_dim(self):
        """embed_dim must be 512 to match the U-Net decoder's C5 channels."""
        with pytest.raises(AssertionError):
            HybridCNNViT(embed_dim=384)

    def test_backward_flows(self):
        h = HybridCNNViT(image_size=(64, 64))
        x = torch.randn(1, 3, 64, 64, requires_grad=True)
        c3, c4, c5 = h(x)
        (c3.sum() + c4.sum() + c5.sum()).backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()


class TestHybridInUNet:

    def test_drop_in_replacement_for_resnet(self):
        """UNet(HybridCNNViT) must build and produce (B, num_classes, H, W)."""
        net = UNet(HybridCNNViT(image_size=(64, 64)), num_classes=5)
        out = net(torch.randn(2, 3, 64, 64))
        assert out.shape == (2, 5, 64, 64)

    def test_unet_hybrid_backward(self):
        """Gradients must reach the decoder + classifier through the hybrid backbone."""
        net = UNet(HybridCNNViT(image_size=(64, 64)), num_classes=5)
        net(torch.randn(1, 3, 64, 64)).sum().backward()
        assert all(p.grad is not None for p in net.classifier.parameters())
