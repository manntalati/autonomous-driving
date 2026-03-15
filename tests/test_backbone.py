"""
Unit tests for Phase 1 backbone models.
These tests use synthetic tensors — no nuScenes data required.
Safe to run in CI without GPU.
"""

import pytest
import torch


# ── LinearClassifier ──────────────────────────────────────────────────────────

class TestLinearClassifier:
    def setup_method(self):
        from models.backbone.linear_classifier import LinearClassifier
        self.model = LinearClassifier(input_dim=3 * 64 * 64, num_classes=3)

    def test_output_shape(self):
        x = torch.randn(4, 3, 64, 64)
        logits = self.model(x)
        assert logits.shape == (4, 3)

    def test_single_item_batch(self):
        x = torch.randn(1, 3, 64, 64)
        logits = self.model(x)
        assert logits.shape == (1, 3)

    def test_no_softmax_in_output(self):
        """Logits should not be in [0,1] — CrossEntropyLoss expects raw scores."""
        x = torch.randn(8, 3, 64, 64)
        logits = self.model(x)
        # If softmax were applied, all values would be in (0,1) and sum to 1
        assert not (logits > 0).all() or not (logits < 1).all()

    def test_gradient_flows(self):
        x = torch.randn(4, 3, 64, 64)
        logits = self.model(x)
        loss = logits.sum()
        loss.backward()
        for name, param in self.model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


# ── MLP ───────────────────────────────────────────────────────────────────────

class TestMLP:
    def setup_method(self):
        from models.backbone.mlp import MLP
        self.model = MLP(input_dim=3 * 64 * 64, hidden_dim=128, num_classes=3)

    def test_output_shape(self):
        x = torch.randn(4, 3, 64, 64)
        logits = self.model(x)
        assert logits.shape == (4, 3)

    def test_gradient_flows(self):
        x = torch.randn(4, 3, 64, 64)
        logits = self.model(x)
        logits.sum().backward()
        for name, param in self.model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


# ── ConvBlock ─────────────────────────────────────────────────────────────────

class TestConvBlock:
    def test_output_shape_preserves_spatial(self):
        from models.backbone.resnet import ConvBlock
        block = ConvBlock(3, 64)
        x = torch.randn(2, 3, 56, 56)
        out = block(x)
        assert out.shape == (2, 64, 56, 56)

    def test_stride_halves_spatial(self):
        from models.backbone.resnet import ConvBlock
        block = ConvBlock(64, 128, stride=2)
        x = torch.randn(2, 64, 56, 56)
        out = block(x)
        assert out.shape == (2, 128, 28, 28)


# ── ResidualBlock ─────────────────────────────────────────────────────────────

class TestResidualBlock:
    def test_identity_shortcut(self):
        """Same channels + stride=1 → identity shortcut."""
        from models.backbone.resnet import ResidualBlock
        block = ResidualBlock(64, 64, stride=1)
        x = torch.randn(2, 64, 28, 28)
        out = block(x)
        assert out.shape == (2, 64, 28, 28)

    def test_projection_shortcut_stride(self):
        """stride=2 + channel change → projected shortcut."""
        from models.backbone.resnet import ResidualBlock
        block = ResidualBlock(64, 128, stride=2)
        x = torch.randn(2, 64, 28, 28)
        out = block(x)
        assert out.shape == (2, 128, 14, 14)

    def test_projection_shortcut_channels_only(self):
        """stride=1 + channel change → projected shortcut."""
        from models.backbone.resnet import ResidualBlock
        block = ResidualBlock(64, 128, stride=1)
        x = torch.randn(2, 64, 28, 28)
        out = block(x)
        assert out.shape == (2, 128, 28, 28)

    def test_gradient_flows(self):
        from models.backbone.resnet import ResidualBlock
        block = ResidualBlock(64, 128, stride=2)
        x = torch.randn(2, 64, 28, 28)
        out = block(x)
        out.sum().backward()
        for name, param in block.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


# ── ResNetBackbone ────────────────────────────────────────────────────────────

class TestResNetBackbone:
    @pytest.fixture
    def small_input(self):
        torch.manual_seed(42)
        return torch.randn(2, 3, 112, 200)  # 1/4 of full res for speed

    def test_classifier_output_shape(self, small_input):
        from models.backbone.resnet import ResNetBackbone
        model = ResNetBackbone(num_classes=3)
        logits = model(small_input)
        assert logits.shape == (2, 3)

    def test_feature_map_output(self, small_input):
        """Backbone without classifier returns (C3, C4, C5) tuple."""
        from models.backbone.resnet import ResNetBackbone
        model = ResNetBackbone(num_classes=None)
        c3, c4, c5 = model(small_input)
        # Check channel dims; spatial dims depend on input
        assert c3.shape[1] == 128
        assert c4.shape[1] == 256
        assert c5.shape[1] == 512

    def test_c4_half_spatial_of_c3(self, small_input):
        from models.backbone.resnet import ResNetBackbone
        import math
        model = ResNetBackbone(num_classes=None)
        c3, c4, c5 = model(small_input)
        # stride-2 conv output = ceil(input / 2), not floor
        assert c4.shape[2] == math.ceil(c3.shape[2] / 2)
        assert c4.shape[3] == math.ceil(c3.shape[3] / 2)

    def test_c5_half_spatial_of_c4(self, small_input):
        from models.backbone.resnet import ResNetBackbone
        import math
        model = ResNetBackbone(num_classes=None)
        c3, c4, c5 = model(small_input)
        assert c5.shape[2] == math.ceil(c4.shape[2] / 2)
        assert c5.shape[3] == math.ceil(c4.shape[3] / 2)

    def test_gradient_flows_classifier(self, small_input):
        from models.backbone.resnet import ResNetBackbone
        model = ResNetBackbone(num_classes=3)
        logits = model(small_input)
        logits.sum().backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_train_eval_mode_toggle(self, small_input):
        """BN behaves differently in train vs eval — both should run without error."""
        from models.backbone.resnet import ResNetBackbone
        model = ResNetBackbone(num_classes=3)
        model.train()
        model(small_input)
        model.eval()
        with torch.no_grad():
            model(small_input)
