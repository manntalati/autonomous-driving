from __future__ import annotations
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_classes: int = 3, dropout: float = 0.3) -> None:
        """
        3-layer MLP: flatten → Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear.
        Args: input_dim — flattened input size; hidden_dim — width of both hidden layers;
              num_classes — output logits; dropout — dropout probability after each hidden layer.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: x — (B, C, H, W) or any shape; flattened internally.
        Returns: (B, num_classes) raw logits.
        """
        return self.net(x)
