"""MLP model for network intrusion detection (PyTorch)."""
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Simple feed-forward network for tabular NIDS features."""

    def __init__(self, n_features: int, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),  # 2-class: [benign, attack]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
