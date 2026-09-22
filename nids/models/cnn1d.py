"""CNN1D model for network intrusion detection (PyTorch).

Treats the feature vector as a 1D signal and applies convolutions
to capture local feature interactions.
"""
import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """1D CNN for tabular NIDS features.

    Architecture: Conv1d -> ReLU -> Conv1d -> ReLU -> GlobalAvgPool -> FC -> FC
    """

    def __init__(self, n_features: int, channels: list = [32, 64], n_classes: int = 2):
        super().__init__()
        layers = []
        in_ch = 1
        for ch in channels:
            layers.extend([
                nn.Conv1d(in_ch, ch, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ])
            in_ch = ch
        self.conv = nn.Sequential(*layers)

        # Compute feature dim after conv + pooling
        feat_len = n_features
        for _ in channels:
            feat_len = feat_len // 2
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_ch * 1, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features) -> (batch, 1, n_features)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.global_pool(x).squeeze(-1)  # (batch, channels)
        return self.fc(x)
