"""
Vision Transformer components, built from scratch (Phase 4).

Pieces:
  - PatchEmbedding        — image/feature-map → sequence of patch tokens
  - PositionalEncoding    — learned positional embedding added to tokens
  - MultiHeadSelfAttention — scaled dot-product attention across heads
  - TransformerEncoderBlock — pre-norm MSA + MLP with residual connections
  - ViT                   — full encoder; classifier head OR feature-map output
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Split an image (or feature map) into non-overlapping patches and linearly
    embed each patch into a token vector.

    A Conv2d with kernel_size == stride == patch_size does this in one op:
    each conv window is one patch, and the conv projects it to embed_dim.
    """

    def __init__(self, in_channels: int = 3, patch_size: int = 16, embed_dim: int = 384) -> None:
        """
        Args:
          in_channels — channels of the input (3 for RGB, or C for a CNN feature map).
          patch_size  — side length of each square patch (also the conv stride).
          embed_dim   — token dimensionality.
        """
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: x — (B, in_channels, H, W). H and W must be divisible by patch_size.
        Returns: (B, num_patches, embed_dim) token sequence, num_patches = (H/p)·(W/p).
        """
        return self.proj(x).flatten(2).transpose(1, 2)


class PositionalEncoding(nn.Module):
    """
    Learned positional embedding. Self-attention is permutation-invariant, so
    without this the model cannot tell where a patch came from.
    """

    def __init__(self, num_patches: int, embed_dim: int) -> None:
        """
        Args: num_patches — fixed token count (grid is fixed by image + patch size);
              embed_dim — token dimensionality.
        """
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: x — (B, num_patches, embed_dim) tokens.
        Returns: (B, num_patches, embed_dim) — x + positional embedding (broadcast over batch).
        """
        return x + self.pos_embed


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head scaled dot-product self-attention.
    Each head attends in a (embed_dim / num_heads)-dim subspace; heads are
    concatenated and projected back to embed_dim.
    """

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        """
        Args: embed_dim — token dim; num_heads — number of attention heads
              (embed_dim must be divisible by num_heads).
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)   # fused Q, K, V projection
        self.proj = nn.Linear(embed_dim, embed_dim)      # output projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: x — (B, N, embed_dim) token sequence.
        Returns: (B, N, embed_dim).
        """
        B, N, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class TransformerEncoderBlock(nn.Module):
    """
    One pre-norm Transformer encoder block:
        x = x + MSA(LayerNorm(x))
        x = x + MLP(LayerNorm(x))
    Pre-norm (norm before the sublayer) keeps gradients stable in deep stacks.
    """

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        """
        Args: embed_dim — token dim; num_heads — attention heads;
              mlp_ratio — hidden dim of the MLP as a multiple of embed_dim;
              dropout — dropout prob inside the MLP.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: x — (B, N, embed_dim).
        Returns: (B, N, embed_dim).
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    """
    Vision Transformer encoder.
    With num_classes set → mean-pools tokens and returns (B, num_classes) logits.
    With num_classes None → returns the token grid as a (B, embed_dim, gh, gw)
    feature map, so it can act as a backbone.
    """

    def __init__(self, in_channels: int = 3, patch_size: int = 16, embed_dim: int = 384, depth: int = 6, num_heads: int = 6, mlp_ratio: float = 4.0, image_size: Tuple[int, int] = (448, 800), num_classes: Optional[int] = None) -> None:
        """
        Args:
          in_channels/patch_size/embed_dim — see PatchEmbedding.
          depth — number of TransformerEncoderBlocks.
          num_heads/mlp_ratio — per-block attention/MLP config.
          image_size — (H, W) the model is built for; fixes the patch grid.
          num_classes — set for classification; None for feature-map (backbone) output.
        """
        super().__init__()
        grid_h = image_size[0] // patch_size
        grid_w = image_size[1] // patch_size
        self.grid = (grid_h, grid_w)
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim)
        self.pos_enc = PositionalEncoding(grid_h * grid_w, embed_dim)
        self.blocks = nn.ModuleList(
            [TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        if num_classes is not None:
            self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: x — (B, in_channels, H, W) matching the configured image_size.
        Returns:
          (B, num_classes) logits if num_classes set;
          else (B, embed_dim, grid_h, grid_w) feature map.
        """
        tokens = self.patch_embed(x)
        tokens = self.pos_enc(tokens) # have to positional encode since its permutation invariant
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        if self.num_classes is not None:
            pooled = tokens.mean(dim=1)
            return self.head(pooled)
        else:
            B, N, D = tokens.shape
            return tokens.transpose(1, 2).reshape(B, D, self.grid[0], self.grid[1])
