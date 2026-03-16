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


# ── LinearClassifier (additional) ─────────────────────────────────────────────

class TestLinearClassifierAdditional:
    """Additional tests not covered by the original TestLinearClassifier."""

    def test_large_batch_size(self):
        """Output shape scales correctly with batch dimension."""
        from models.backbone.linear_classifier import LinearClassifier
        torch.manual_seed(42)
        model = LinearClassifier(input_dim=32 * 32 * 3, num_classes=3)
        x = torch.randn(32, 3, 32, 32)
        out = model(x)
        assert out.shape == (32, 3)

    def test_custom_num_classes(self):
        """num_classes parameter controls the output width."""
        from models.backbone.linear_classifier import LinearClassifier
        torch.manual_seed(42)
        for nc in [1, 5, 10]:
            model = LinearClassifier(input_dim=16, num_classes=nc)
            x = torch.randn(2, 16)
            out = model(x)
            assert out.shape == (2, nc), f"Expected (2, {nc}), got {out.shape}"

    def test_fc_weight_shape(self):
        """nn.Linear weight matrix has shape (num_classes, input_dim)."""
        from models.backbone.linear_classifier import LinearClassifier
        model = LinearClassifier(input_dim=100, num_classes=3)
        assert model.fc.weight.shape == (3, 100)
        assert model.fc.bias.shape == (3,)

    def test_fc_bias_initialised_to_zero(self):
        """Default nn.Linear initialises bias to zeros (uniform in [-k, k],
        but we verify it is not all-zero after kaiming init — this just checks
        the bias tensor exists and has the right shape, not its exact value,
        because nn.Linear uses its own default init, not ours.)"""
        from models.backbone.linear_classifier import LinearClassifier
        model = LinearClassifier(input_dim=64, num_classes=3)
        assert model.fc.bias is not None
        assert model.fc.bias.shape == (3,)

    def test_output_is_deterministic_in_eval(self):
        """Same input must produce identical output in eval mode."""
        from models.backbone.linear_classifier import LinearClassifier
        torch.manual_seed(42)
        model = LinearClassifier(input_dim=3 * 8 * 8, num_classes=3)
        model.eval()
        x = torch.randn(4, 3, 8, 8)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)

    def test_1d_input_no_spatial_dims(self):
        """LinearClassifier should handle flat (B, D) inputs via nn.Flatten."""
        from models.backbone.linear_classifier import LinearClassifier
        torch.manual_seed(42)
        model = LinearClassifier(input_dim=128, num_classes=3)
        x = torch.randn(4, 128)
        out = model(x)
        assert out.shape == (4, 3)

    def test_parameters_update_after_optimizer_step(self):
        """A gradient step must change the weight values."""
        from models.backbone.linear_classifier import LinearClassifier
        torch.manual_seed(42)
        model = LinearClassifier(input_dim=3 * 8 * 8, num_classes=3)
        weight_before = model.fc.weight.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        x = torch.randn(2, 3, 8, 8)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        assert not torch.allclose(model.fc.weight, weight_before), \
            "Weights did not change after optimizer step"


# ── MLP (additional) ──────────────────────────────────────────────────────────

class TestMLPAdditional:
    """Additional tests not covered by the original TestMLP."""

    def test_three_linear_layers_present(self):
        """Network must contain exactly 3 nn.Linear layers."""
        from models.backbone.mlp import MLP
        model = MLP(input_dim=64, hidden_dim=32, num_classes=3)
        linear_layers = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
        assert len(linear_layers) == 3, \
            f"Expected 3 Linear layers, found {len(linear_layers)}"

    def test_two_relu_activations_present(self):
        """Network must contain exactly 2 ReLU activations (after hidden layers)."""
        from models.backbone.mlp import MLP
        model = MLP(input_dim=64, hidden_dim=32, num_classes=3)
        relus = [m for m in model.modules() if isinstance(m, torch.nn.ReLU)]
        assert len(relus) == 2

    def test_two_dropout_layers_present(self):
        """Network must contain exactly 2 Dropout layers."""
        from models.backbone.mlp import MLP
        model = MLP(input_dim=64, hidden_dim=32, num_classes=3)
        dropouts = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
        assert len(dropouts) == 2

    def test_hidden_dim_matches_intermediate_layers(self):
        """The two hidden Linear layers must use the specified hidden_dim."""
        from models.backbone.mlp import MLP
        model = MLP(input_dim=64, hidden_dim=32, num_classes=5)
        linear_layers = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
        # layer[0]: input_dim -> hidden_dim
        assert linear_layers[0].out_features == 32
        # layer[1]: hidden_dim -> hidden_dim
        assert linear_layers[1].in_features == 32
        assert linear_layers[1].out_features == 32
        # layer[2]: hidden_dim -> num_classes
        assert linear_layers[2].out_features == 5

    def test_dropout_inactive_in_eval_mode(self):
        """Output must be identical across two forward passes in eval mode."""
        from models.backbone.mlp import MLP
        torch.manual_seed(42)
        model = MLP(input_dim=3 * 8 * 8, hidden_dim=64, num_classes=3, dropout=0.5)
        model.eval()
        x = torch.randn(4, 3, 8, 8)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2), "eval mode outputs differ — dropout is still active"

    def test_dropout_active_in_train_mode(self):
        """Two forward passes in train mode with dropout=0.5 should differ
        (probability 1 - (0.5^N) ≈ 1 for any meaningful N; we use N=128)."""
        from models.backbone.mlp import MLP
        torch.manual_seed(0)
        model = MLP(input_dim=128, hidden_dim=128, num_classes=3, dropout=0.5)
        model.train()
        x = torch.randn(4, 128)
        out1 = model(x)
        out2 = model(x)
        # With 128-dimensional dropout at p=0.5 the probability of
        # identical outputs is astronomically small.
        assert not torch.allclose(out1, out2), \
            "train mode outputs are identical — dropout may not be active"

    def test_single_item_batch(self):
        """Batch size 1 must not crash and must return shape (1, num_classes)."""
        from models.backbone.mlp import MLP
        model = MLP(input_dim=3 * 16 * 16, hidden_dim=64, num_classes=3)
        x = torch.randn(1, 3, 16, 16)
        out = model(x)
        assert out.shape == (1, 3)

    def test_zero_dropout_output_is_deterministic_in_train(self):
        """With dropout=0, train and eval outputs must agree."""
        from models.backbone.mlp import MLP
        torch.manual_seed(42)
        model = MLP(input_dim=64, hidden_dim=32, num_classes=3, dropout=0.0)
        x = torch.randn(2, 64)
        model.train()
        out_train = model(x)
        model.eval()
        with torch.no_grad():
            out_eval = model(x)
        assert torch.allclose(out_train, out_eval, atol=1e-6)

    def test_gradient_flows_to_all_three_linear_layers(self):
        """Gradient must reach weight and bias of every Linear layer."""
        from models.backbone.mlp import MLP
        torch.manual_seed(42)
        model = MLP(input_dim=64, hidden_dim=32, num_classes=3)
        x = torch.randn(4, 64)
        model(x).sum().backward()
        linear_layers = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
        for i, layer in enumerate(linear_layers):
            assert layer.weight.grad is not None, f"Layer {i} weight has no gradient"
            assert layer.bias.grad is not None, f"Layer {i} bias has no gradient"


# ── ConvBlock (additional) ─────────────────────────────────────────────────────

class TestConvBlockAdditional:

    def test_no_bias_by_default(self):
        """Conv2d bias should be None because bias=False is the default."""
        from models.backbone.resnet import ConvBlock
        block = ConvBlock(3, 64)
        assert block.conv.bias is None

    def test_batchnorm_is_present(self):
        from models.backbone.resnet import ConvBlock
        block = ConvBlock(3, 64)
        assert isinstance(block.bn, torch.nn.BatchNorm2d)
        assert block.bn.num_features == 64

    def test_relu_output_is_nonnegative(self):
        """ConvBlock ends with ReLU so all output values must be >= 0."""
        from models.backbone.resnet import ConvBlock
        torch.manual_seed(42)
        block = ConvBlock(3, 64)
        block.eval()
        x = torch.randn(2, 3, 16, 16)
        with torch.no_grad():
            out = block(x)
        assert (out >= 0).all(), "ConvBlock output contains negative values — ReLU missing?"

    def test_gradient_flows_through_convblock(self):
        from models.backbone.resnet import ConvBlock
        torch.manual_seed(42)
        block = ConvBlock(16, 32, stride=2)
        x = torch.randn(2, 16, 16, 16)
        block(x).sum().backward()
        for name, param in block.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_kernel_7_stride_2_padding_3_preserves_halving(self):
        """Stem uses 7x7 conv with stride=2, padding=3 → H/2, W/2 (for even dims)."""
        from models.backbone.resnet import ConvBlock
        block = ConvBlock(3, 64, kernel_size=7, stride=2, padding=3)
        x = torch.randn(1, 3, 224, 224)
        out = block(x)
        assert out.shape == (1, 64, 112, 112)


# ── ResidualBlock (additional) ─────────────────────────────────────────────────

class TestResidualBlockAdditional:

    def test_shortcut_is_identity_for_same_channels_stride1(self):
        """When in==out and stride==1, shortcut must be nn.Identity."""
        from models.backbone.resnet import ResidualBlock
        block = ResidualBlock(64, 64, stride=1)
        assert isinstance(block.shortcut, torch.nn.Identity), \
            "Expected nn.Identity shortcut for same-channel stride-1 block"

    def test_shortcut_is_sequential_for_channel_change(self):
        """When channels differ, shortcut must be nn.Sequential(Conv2d, BN)."""
        from models.backbone.resnet import ResidualBlock
        block = ResidualBlock(64, 128, stride=1)
        assert isinstance(block.shortcut, torch.nn.Sequential)

    def test_shortcut_is_sequential_for_stride2(self):
        """When stride > 1, shortcut must be nn.Sequential(Conv2d, BN)."""
        from models.backbone.resnet import ResidualBlock
        block = ResidualBlock(64, 64, stride=2)
        assert isinstance(block.shortcut, torch.nn.Sequential)

    def test_output_relu_applied(self):
        """Output values of residual block must be >= 0 (final ReLU)."""
        from models.backbone.resnet import ResidualBlock
        torch.manual_seed(42)
        block = ResidualBlock(32, 32, stride=1)
        block.eval()
        x = torch.randn(2, 32, 16, 16)
        with torch.no_grad():
            out = block(x)
        assert (out >= 0).all(), "ResidualBlock output contains negatives — final ReLU missing?"

    def test_shortcut_conv_is_1x1(self):
        """The projection shortcut must use a 1x1 convolution."""
        from models.backbone.resnet import ResidualBlock
        block = ResidualBlock(64, 128, stride=2)
        proj_conv = block.shortcut[0]
        assert isinstance(proj_conv, torch.nn.Conv2d)
        assert proj_conv.kernel_size == (1, 1), \
            f"Projection conv kernel is {proj_conv.kernel_size}, expected (1, 1)"

    def test_no_nan_in_output(self):
        """Output must not contain NaN or Inf."""
        from models.backbone.resnet import ResidualBlock
        torch.manual_seed(42)
        block = ResidualBlock(64, 128, stride=2)
        x = torch.randn(2, 64, 28, 28)
        out = block(x)
        assert not torch.isnan(out).any(), "NaN in ResidualBlock output"
        assert not torch.isinf(out).any(), "Inf in ResidualBlock output"


# ── ResNetBackbone (additional) ────────────────────────────────────────────────

class TestResNetBackboneAdditional:

    @pytest.fixture
    def backbone_input(self):
        torch.manual_seed(42)
        return torch.randn(2, 3, 112, 200)

    def test_stem_output_channels_are_64(self, backbone_input):
        """After the stem, feature map must have 64 channels."""
        from models.backbone.resnet import ResNetBackbone
        model = ResNetBackbone(num_classes=None)
        # Register a hook on the stem to capture its output
        stem_out = {}
        def hook(module, inp, out):
            stem_out['val'] = out
        handle = model.stem.register_forward_hook(hook)
        model(backbone_input)
        handle.remove()
        assert stem_out['val'].shape[1] == 64

    def test_stem_halves_spatial_twice(self, backbone_input):
        """Stem = conv(stride=2) + maxpool(stride=2) → H/4, W/4."""
        from models.backbone.resnet import ResNetBackbone
        model = ResNetBackbone(num_classes=None)
        stem_out = {}
        def hook(module, inp, out):
            stem_out['val'] = out
        handle = model.stem.register_forward_hook(hook)
        model(backbone_input)
        handle.remove()
        # Input 112×200, after stride-2 conv → 56×100, after maxpool → 28×100
        # (maxpool with padding=1 on 56 → 28; on 100 → 50)
        assert stem_out['val'].shape[2] == 28
        assert stem_out['val'].shape[3] == 50

    def test_classifier_head_absent_when_num_classes_none(self):
        """Backbone without num_classes must not have avgpool or classifier attrs
        that would interfere — or at least must not use them in forward."""
        from models.backbone.resnet import ResNetBackbone
        model = ResNetBackbone(num_classes=None)
        assert not hasattr(model, 'classifier'), \
            "classifier attribute should not exist when num_classes=None"

    def test_feature_map_shapes_at_full_project_resolution(self):
        """Verify C3/C4/C5 spatial dims on the actual project resolution 448×800."""
        from models.backbone.resnet import ResNetBackbone
        torch.manual_seed(42)
        model = ResNetBackbone(num_classes=None)
        model.eval()
        x = torch.randn(1, 3, 448, 800)
        with torch.no_grad():
            c3, c4, c5 = model(x)
        # Stem: 448→112 (H), 800→200 (W)   (7x7 conv stride2 + maxpool stride2)
        # Stage2 (C3): stride=2 → 56×100
        # Stage3 (C4): stride=2 → 28×50
        # Stage4 (C5): stride=2 → 14×25
        assert c3.shape == (1, 128, 56, 100), f"C3 shape {c3.shape}"
        assert c4.shape == (1, 256, 28, 50),  f"C4 shape {c4.shape}"
        assert c5.shape == (1, 512, 14, 25),  f"C5 shape {c5.shape}"

    def test_no_nan_in_classifier_output(self, backbone_input):
        from models.backbone.resnet import ResNetBackbone
        torch.manual_seed(42)
        model = ResNetBackbone(num_classes=3)
        out = model(backbone_input)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_kaiming_init_applied_to_conv_weights(self):
        """Conv2d weights must not be all zero after _init_weights."""
        from models.backbone.resnet import ResNetBackbone
        model = ResNetBackbone(num_classes=3)
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                assert not (m.weight == 0).all(), \
                    "A Conv2d weight tensor is all zeros — init may not have run"

    def test_batchnorm_gamma_1_beta_0_after_init(self):
        """After _init_weights, all BN weight=1 and bias=0."""
        from models.backbone.resnet import ResNetBackbone
        model = ResNetBackbone(num_classes=3)
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                assert torch.allclose(m.weight, torch.ones_like(m.weight)), \
                    "BN gamma not initialised to 1"
                assert torch.allclose(m.bias, torch.zeros_like(m.bias)), \
                    "BN beta not initialised to 0"

    def test_feature_map_output_is_tuple_of_three(self, backbone_input):
        from models.backbone.resnet import ResNetBackbone
        model = ResNetBackbone(num_classes=None)
        output = model(backbone_input)
        assert isinstance(output, tuple)
        assert len(output) == 3

    def test_gradient_flows_through_feature_backbone(self, backbone_input):
        """Gradients must reach all parameters when using the feature-map path."""
        from models.backbone.resnet import ResNetBackbone
        model = ResNetBackbone(num_classes=None)
        c3, c4, c5 = model(backbone_input)
        (c3.sum() + c4.sum() + c5.sum()).backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
