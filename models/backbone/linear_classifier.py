from __future__ import annotations
import torch
import torch.nn as nn

class LinearClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 3) -> None:
        """
        Single linear layer classifier (flatten → fc).
        Args: input_dim — total flattened input size; num_classes — output logits count.
        """
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: x — (B, C, H, W) or any shape; flattened internally.
        Returns: (B, num_classes) raw logits.
        """
        return self.fc(self.flatten(x))
