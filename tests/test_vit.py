"""
Unit tests for models/backbone/vit.py — from-scratch Vision Transformer.

Verify:
  - PatchEmbedding turns an image (or feature map) into the right token count
  - PositionalEncoding is a learnable, shape-preserving add
  - MultiHeadSelfAttention is shape-preserving and differentiable
  - TransformerEncoderBlock is shape-preserving with working residuals
  - ViT runs in both classification and feature-map (backbone) modes

Random tensors only — no nuScenes data, no GPU, no weight downloads.
"""

import pytest
import torch

from models.backbone.vit import (
    PatchEmbedding,
    PositionalEncoding,
    MultiHeadSelfAttention,
    TransformerEncoderBlock,
    ViT,
)


class TestPatchEmbedding:

    def test_token_count_and_dim(self):
        """A 64x96 image at patch 16 → a 4x6 grid = 24 tokens of width embed_dim."""
        pe = PatchEmbedding(in_channels=3, patch_size=16, embed_dim=64)
        out = pe(torch.randn(2, 3, 64, 96))
        assert out.shape == (2, 24, 64)

    def test_accepts_feature_map_input(self):
        """Must also embed a CNN feature map (in_channels != 3) — used by the hybrid."""
        pe = PatchEmbedding(in_channels=256, patch_size=2, embed_dim=512)
        out = pe(torch.randn(1, 256, 8, 8))
        assert out.shape == (1, 16, 512)

    def test_differentiable(self):
        pe = PatchEmbedding(3, 16, 64)
        x = torch.randn(1, 3, 64, 64, requires_grad=True)
        pe(x).sum().backward()
        assert x.grad is not None


class TestPositionalEncoding:

    def test_shape_preserving(self):
        pos = PositionalEncoding(num_patches=16, embed_dim=64)
        x = torch.randn(2, 16, 64)
        assert pos(x).shape == x.shape

    def test_pos_embed_is_learnable(self):
        pos = PositionalEncoding(num_patches=16, embed_dim=64)
        assert isinstance(pos.pos_embed, torch.nn.Parameter)
        assert pos.pos_embed.requires_grad

    def test_actually_adds_position(self):
        """Output must differ from input — pos_embed is non-zero (trunc_normal init)."""
        pos = PositionalEncoding(num_patches=16, embed_dim=64)
        x = torch.zeros(1, 16, 64)
        assert not torch.allclose(pos(x), x)


class TestMultiHeadSelfAttention:

    def test_shape_preserving(self):
        mhsa = MultiHeadSelfAttention(embed_dim=64, num_heads=8)
        x = torch.randn(2, 24, 64)
        assert mhsa(x).shape == x.shape

    def test_differentiable(self):
        mhsa = MultiHeadSelfAttention(64, 8)
        x = torch.randn(2, 24, 64, requires_grad=True)
        mhsa(x).sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_rejects_indivisible_head_count(self):
        """embed_dim must be divisible by num_heads."""
        with pytest.raises(AssertionError):
            MultiHeadSelfAttention(embed_dim=64, num_heads=7)

    def test_single_head(self):
        mhsa = MultiHeadSelfAttention(embed_dim=64, num_heads=1)
        assert mhsa(torch.randn(1, 10, 64)).shape == (1, 10, 64)


class TestTransformerEncoderBlock:

    def test_shape_preserving(self):
        blk = TransformerEncoderBlock(embed_dim=64, num_heads=8)
        x = torch.randn(2, 24, 64)
        assert blk(x).shape == x.shape

    def test_residual_changes_input(self):
        blk = TransformerEncoderBlock(64, 8)
        x = torch.randn(2, 24, 64)
        assert not torch.allclose(blk(x), x)

    def test_differentiable(self):
        blk = TransformerEncoderBlock(64, 8)
        x = torch.randn(2, 24, 64, requires_grad=True)
        blk(x).sum().backward()
        assert x.grad is not None


class TestViT:

    def test_classification_output_shape(self):
        """num_classes set → (B, num_classes) logits."""
        vit = ViT(patch_size=16, embed_dim=64, depth=2, num_heads=8,
                  image_size=(64, 64), num_classes=10)
        out = vit(torch.randn(2, 3, 64, 64))
        assert out.shape == (2, 10)

    def test_backbone_feature_map_shape(self):
        """num_classes=None → (B, embed_dim, grid_h, grid_w) feature map."""
        vit = ViT(patch_size=16, embed_dim=64, depth=2, num_heads=8,
                  image_size=(64, 96), num_classes=None)
        out = vit(torch.randn(2, 3, 64, 96))
        assert out.shape == (2, 64, 4, 6)

    def test_backward_flows(self):
        vit = ViT(patch_size=16, embed_dim=64, depth=2, num_heads=8,
                  image_size=(64, 64), num_classes=5)
        x = torch.randn(1, 3, 64, 64, requires_grad=True)
        vit(x).sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_classifier_head_only_when_num_classes_set(self):
        assert hasattr(ViT(image_size=(64, 64), patch_size=16, num_classes=7), "head")
        assert not hasattr(ViT(image_size=(64, 64), patch_size=16, num_classes=None), "head")

    def test_eval_mode_deterministic(self):
        vit = ViT(patch_size=16, embed_dim=64, depth=2, num_heads=8,
                  image_size=(64, 64), num_classes=5).eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            assert torch.allclose(vit(x), vit(x))
