"""Self-attention Transformer used by the DMA-PULSE pipeline."""

from __future__ import annotations

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Learned positional representation for short activity sequences."""

    def __init__(self, sequence_length: int, d_model: int) -> None:
        super().__init__()
        self.position = nn.Parameter(torch.zeros(1, sequence_length, d_model))
        nn.init.normal_(self.position, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.position[:, : x.size(1)]


class PulseTransformer(nn.Module):
    """Configurable self-attention classifier for PULSE sequences.

    Inputs are shaped [batch, sequence_length, feature_dim]. The default
    sequence length and feature dimension follow the published representation
    of five time steps and eleven final features.
    """

    def __init__(
        self,
        feature_dim: int = 11,
        sequence_length: int = 5,
        num_classes: int = 2,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        self.input_projection = nn.Linear(feature_dim, d_model)
        self.position = PositionalEncoding(sequence_length, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Expected input shaped [batch, sequence, features]")
        x = self.input_projection(x)
        x = self.position(x)
        x = self.encoder(x)
        x = self.norm(x.mean(dim=1))
        return self.classifier(x)
