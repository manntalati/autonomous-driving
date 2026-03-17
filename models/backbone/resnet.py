from __future__ import annotations
from typing import List, Optional, Tuple
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, bias: bool = False) -> None:
        """
        Conv2d → BatchNorm2d → ReLU building block.
        Args: in_channels/out_channels — channel dims; kernel_size/stride/padding — conv params.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: x — (B, in_channels, H, W).
        Returns: (B, out_channels, H', W') after conv+bn+relu.
        """
        return self.relu(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        """
        Two ConvBlocks with a skip connection. Projection shortcut added when stride≠1 or channels change.
        Args: in_channels/out_channels — channel dims; stride — applied to first conv and shortcut.
        """
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, stride=stride)
        self.conv2 = ConvBlock(out_channels, out_channels, stride=1)
        self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride, bias=False), nn.BatchNorm2d(out_channels)) if stride != 1 or in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: x — (B, in_channels, H, W).
        Returns: (B, out_channels, H', W') after residual addition + relu.
        """
        return self.relu(self.conv2(self.conv1(x)) + self.shortcut(x))

def _make_stage(in_channels: int, out_channels: int, num_blocks: int, stride: int = 1) -> nn.Sequential:
    """
    Build one ResNet stage: first block may downsample (stride), remaining blocks use stride=1.
    Args: in_channels/out_channels — channel dims; num_blocks — number of ResidualBlocks; stride — for first block.
    Returns: nn.Sequential of ResidualBlocks.
    """
    layers = []
    layers.append(ResidualBlock(in_channels, out_channels, stride=stride))
    for _ in range(num_blocks - 1):
        layers.append(ResidualBlock(out_channels, out_channels, stride=1))
    return nn.Sequential(*layers)

class ResNetBackbone(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: Optional[int] = None) -> None:
        """
        ResNet-style backbone: stem (7x7 conv + maxpool) → 4 stages → optional classifier head.
        Args: in_channels — input image channels (3 for RGB);
              num_classes — if set, adds avgpool+linear for classification; if None, returns (C3, C4, C5) feature maps.
        """
        super().__init__()
        self.stem = nn.Sequential(
            ConvBlock(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.stage1 = _make_stage(in_channels=64, out_channels=64, num_blocks=2, stride=1)
        self.stage2 = _make_stage(in_channels=64, out_channels=128, num_blocks=2, stride=2)
        self.stage3 = _make_stage(in_channels=128, out_channels=256, num_blocks=2, stride=2)
        self.stage4 = _make_stage(in_channels=256, out_channels=512, num_blocks=2, stride=2)

        self.num_classes = num_classes
        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.classifier = nn.Linear(512, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming init for conv layers; constant init for BN weight/bias."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor | Tuple[torch.Tensor, ...]:
        """
        Args: x — (B, 3, H, W) input image.
        Returns: (B, num_classes) logits if num_classes set;
                 else tuple (C3, C4, C5) — multi-scale feature maps at 1/8, 1/16, 1/32 of input resolution.
        """
        c3 = self.stage2(self.stage1(self.stem(x)))
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        if self.num_classes is not None:
            return self.classifier(torch.flatten(self.avgpool(c5), 1))
        else:
            return (c3, c4, c5)
