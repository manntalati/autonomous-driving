"""
Hybrid CNN-ViT backbone (Phase 4).

Motivation: CNNs have a strong local inductive bias (locality, translation
equivariance) and are cheap at high resolution; Transformers reason globally
but are data-hungry and quadratic in token count. A hybrid uses the CNN to
extract local features cheaply, then a small ViT encoder for global reasoning
on the (already downsampled, so few-token) deepest stage.

This backbone returns (C3, C4, C5) with channels 128 / 256 / 512 at strides
8 / 16 / 32 — identical to ResNetBackbone — so it is a DROP-IN replacement
inside the Phase 3 UNet. The mIoU comparison is then: same U-Net + same loss
+ same train_seg.py, only the backbone differs (selected via config).

  - C3 (stride 8,  128 ch) — ResNet stem + stage1 + stage2   (CNN, local)
  - C4 (stride 16, 256 ch) — ResNet stage3                    (CNN, local)
  - C5 (stride 32, 512 ch) — ViT encoder over a patch-embedded C4 (global)
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn

from models.backbone.resnet import ResNetBackbone
from models.backbone.vit import PatchEmbedding, PositionalEncoding, TransformerEncoderBlock


class HybridCNNViT(nn.Module):
    """
    CNN front (ResNet stem + stages 1-3) for local C3/C4 features, then a ViT
    encoder that turns C4 into a globally-reasoned C5. Output matches
    ResNetBackbone's (C3, C4, C5) contract so UNet consumes it unchanged.

    Note: embed_dim MUST be 512 — UNet's decoder hardcodes a 512-channel C5.
    """

    def __init__(self, embed_dim: int = 512, depth: int = 4, num_heads: int = 8, mlp_ratio: float = 4.0, image_size: Tuple[int, int] = (448, 800), num_classes: Optional[int] = None) -> None:
        """
        Args:
          embed_dim — C5 channel count; must be 512 for drop-in UNet compatibility.
          depth — number of TransformerEncoderBlocks in the ViT stage.
          num_heads/mlp_ratio — per-block attention/MLP config.
          image_size — (H, W) the model is built for; fixes the C5 token grid.
          num_classes — kept None: this is a feature-map backbone (UNet asserts None).
        """
        super().__init__()
        assert embed_dim == 512, "embed_dim must be 512 to match UNet's C5 decoder channels"
        self.num_classes = num_classes  # None → feature-map backbone

        # CNN front: reuse ResNetBackbone's stem + stage1-3 (stage4/classifier unused).
        self.cnn = ResNetBackbone()

        # ViT stage: C4 is stride-16 / 256-ch → patch_size 2 takes it to stride-32,
        # embed_dim tokens. Grid is fixed by image_size.
        grid_h = image_size[0] // 32
        grid_w = image_size[1] // 32
        self.grid = (grid_h, grid_w)
        self.patch_embed = PatchEmbedding(in_channels=256, patch_size=2, embed_dim=embed_dim)
        self.pos_enc = PositionalEncoding(grid_h * grid_w, embed_dim)
        self.blocks = nn.ModuleList(
            [TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args: x — (B, 3, H, W) matching the configured image_size.
        Returns: (C3, C4, C5) feature maps —
          C3 (B, 128, H/8,  W/8), C4 (B, 256, H/16, W/16), C5 (B, 512, H/32, W/32).
        """
        c3 = self.cnn.stage2(self.cnn.stage1(self.cnn.stem(x)))
        c4 = self.cnn.stage3(c3)
        tokens = self.patch_embed(c4)
        tokens = self.pos_enc(tokens)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        B, _, D = tokens.shape
        c5 = tokens.transpose(1, 2).reshape(B, D, self.grid[0], self.grid[1])
        return c3, c4, c5

    def load_pretrained(self) -> None:
        """
        Load ImageNet-pretrained weights into the CNN front (stem + all ResNet
        stages). The ViT stage is left at its random init — there is no
        pretrained ViT in this project, and training it from scratch is the
        point of the comparison.
        """
        self.cnn.load_pretrained()
