# src/fusion/fusion.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding (copied from old components.py)."""
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MeanFusion(nn.Module):
    """Average pooling over the frame dimension."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, num_frames, T, D]  -> [B, T, D]
        return x.mean(dim=1)


class MaxFusion(nn.Module):
    """Max pooling over the frame dimension."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, num_frames, T, D]  -> [B, T, D]
        return x.max(dim=1).values


class AttentionFusion(nn.Module):
    """
    Learnable frame‑wise attention.
    Each frame is scored by a small MLP, then softmax‑weighted sum.
    """
    def __init__(self, d_model: int, reduction: int = 8):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(d_model, d_model // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // reduction, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, num_frames, T, D]
        B, num_frames, T, D = x.shape
        # Compute score per frame: aggregate over time
        frame_repr = x.mean(dim=2)          # [B, num_frames, D]
        scores = self.score_net(frame_repr) # [B, num_frames, 1]
        weights = F.softmax(scores, dim=1)  # [B, num_frames, 1]
        # Weighted sum over frames
        weighted = (x * weights.unsqueeze(2)).sum(dim=1)  # [B, T, D]
        return weighted


class TemporalFusionTransformer(nn.Module):
    """
    Cross-frame transformer for fusing multi-frame token features.
    THIS IS THE ORIGINAL VERSION FROM components.py
    Input:  [B, F, T, D]
    Output: [B, F*T, D]  <-- Preserves frame information!
    """
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_frames: int = 5,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_frames = max_frames

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pos_encoding = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.frame_embedding = nn.Embedding(max_frames, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, F, T, D]
        Returns:
            [B, F*T, D]  <-- FLATTENED but with frame identity preserved!
        """
        B, n_frames, T, D = x.shape
        x_flat = x.view(B, n_frames * T, D)

        # Add frame identity so the transformer knows which token came from which frame
        frame_ids = torch.arange(n_frames, device=x.device).unsqueeze(1).repeat(1, T).reshape(1, n_frames * T)
        frame_ids = frame_ids.expand(B, -1)  # [B, F*T]
        x_flat = x_flat + self.frame_embedding(frame_ids)

        x_flat = self.pos_encoding(x_flat)
        fused = self.transformer(x_flat)
        return fused


class ReliabilityWeightedFusion(nn.Module):
    """
    Predict a quality score per frame using a small MLP,
    then perform weighted sum over frames.
    This allows the model to ignore corrupted or low‑quality frames.
    """
    def __init__(self, d_model: int, hidden_dim: int = 128):
        super().__init__()
        self.quality_net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, num_frames, T, D]
        B, num_frames, T, D = x.shape
        # Compute frame representation by averaging over time
        frame_repr = x.mean(dim=2)          # [B, num_frames, D]
        # Predict raw quality scores
        raw_scores = self.quality_net(frame_repr)  # [B, num_frames, 1]
        # Normalise with softmax to get weights
        weights = F.softmax(raw_scores, dim=1)     # [B, num_frames, 1]
        # Weighted sum over frames
        weighted = (x * weights.unsqueeze(2)).sum(dim=1)  # [B, T, D]
        return weighted


# ---------- Factory ----------
def build_fusion(name: str, **kwargs) -> nn.Module:
    """
    Factory for fusion modules.

    Args:
        name: One of 'mean', 'max', 'attention', 'transformer', 'reliability'
        **kwargs: Additional arguments specific to each fusion type.
            - For 'attention' and 'reliability': d_model (required)
            - For 'transformer': d_model (required), nhead, num_layers, dim_feedforward, dropout, max_frames

    Returns:
        nn.Module instance.

    Example:
        fusion = build_fusion('transformer', d_model=512, nhead=8, num_layers=2)
    """
    name = name.lower()
    if name == 'mean':
        return MeanFusion()
    elif name == 'max':
        return MaxFusion()
    elif name == 'attention':
        d_model = kwargs.get('d_model')
        if d_model is None:
            raise ValueError("'attention' fusion requires 'd_model' in kwargs")
        reduction = kwargs.get('reduction', 8)
        return AttentionFusion(d_model, reduction)
    elif name == 'transformer':
        d_model = kwargs.get('d_model')
        if d_model is None:
            raise ValueError("'transformer' fusion requires 'd_model' in kwargs")
        nhead = kwargs.get('nhead', 8)
        num_layers = kwargs.get('num_layers', 2)  # Changed default to match old version
        dim_feedforward = kwargs.get('dim_feedforward', 2048)
        dropout = kwargs.get('dropout', 0.1)
        max_frames = kwargs.get('max_frames', 5)
        return TemporalFusionTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_frames=max_frames
        )
    elif name == 'reliability':
        d_model = kwargs.get('d_model')
        if d_model is None:
            raise ValueError("'reliability' fusion requires 'd_model' in kwargs")
        hidden_dim = kwargs.get('hidden_dim', 128)
        return ReliabilityWeightedFusion(d_model, hidden_dim)
    else:
        raise ValueError(f"Unknown fusion name: {name}. Choose from 'mean', 'max', 'attention', 'transformer', 'reliability'")