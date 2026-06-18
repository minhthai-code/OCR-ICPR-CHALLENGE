"""Multi-frame CRNN architecture (Baseline) with STN and pluggable fusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components import AttentionFusion, CNNBackbone, STNBlock
from src.fusion.fusion import build_fusion


class MultiFrameCRNN(nn.Module):
    """
    Standard CRNN architecture adapted for Multi-frame input with optional STN alignment.
    Pipeline: Input (5 frames) -> [Optional STN] -> CNN Backbone -> Fusion -> BiLSTM -> CTC Head
    """

    def __init__(
        self,
        num_classes: int,
        hidden_size: int = 256,
        rnn_dropout: float = 0.25,
        use_stn: bool = True,
        is_multiframe: bool = True,
        fusion_fn: callable = None,
        fusion_name: str = None,
        **kwargs,
    ):
        super().__init__()
        self.cnn_channels = 512
        self.use_stn = use_stn
        self.is_multiframe = is_multiframe

        # ---- Fusion logic ----
        if not is_multiframe:
            print("CRNN: SINGLE-FRAME mode (using only first frame)")
            self.fusion_fn = None
        elif fusion_name is not None:
            fusion_kwargs = kwargs.copy()
            fusion_kwargs["d_model"] = self.cnn_channels
            self.fusion_module = build_fusion(fusion_name, **fusion_kwargs)
            self.fusion_fn = self._fusion_wrapper
            print(f"CRNN: MULTI-FRAME mode (5 frames) with fusion: {fusion_name}")
        elif fusion_fn is not None:
            self.fusion_fn = fusion_fn
            print("CRNN: MULTI-FRAME mode (5 frames) with custom fusion")
        else:
            self.fusion_fn = None
            print("CRNN: MULTI-FRAME mode (5 frames) using default AttentionFusion")

        # ---- STN alignment ----
        if self.use_stn:
            self.stn = STNBlock(in_channels=3)

        # ---- Feature Extractor ----
        self.backbone = CNNBackbone(out_channels=self.cnn_channels)

        # ---- Default fusion fallback (only used for multi‑frame mode) ----
        self.default_fusion = AttentionFusion(channels=self.cnn_channels)

        # ---- BiLSTM ----
        self.rnn = nn.LSTM(
            input_size=self.cnn_channels,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=rnn_dropout,
        )

        # ---- Head ----
        self.head = nn.Linear(hidden_size * 2, num_classes)

    def _fusion_wrapper(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F, C, W] -> [B, F, W, C] -> fusion -> [B, W, C] -> [B, C, W]
        x_perm = x.permute(0, 1, 3, 2)
        fused = self.fusion_module(x_perm)
        return fused.permute(0, 2, 1)

    def forward(self, x: torch.Tensor, return_frame_logits: bool = False, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [Batch, Frames, 3, H, W]
            return_frame_logits: if True, return (logits, None) for compatibility
        Returns:
            Logits: [Batch, Seq_Len, Num_Classes] (log softmax)
            or (logits, None) if return_frame_logits is True
        """
        B, num_frames, C, H, W = x.shape

        # ----- Single‑frame mode: keep only the first frame -----
        if not self.is_multiframe:
            x = x[:, 0:1, ...]          # [B, 1, C, H, W]
            num_frames = 1

        # ----- STN and backbone (flatten frames) -----
        x_flat = x.view(B * num_frames, C, H, W)

        if self.use_stn:
            theta = self.stn(x_flat)
            grid = F.affine_grid(theta, x_flat.size(), align_corners=False)
            x_aligned = F.grid_sample(x_flat, grid, align_corners=False)
        else:
            x_aligned = x_flat

        features = self.backbone(x_aligned)                     # [B*F, 512, 1, W']
        features = features.view(B, num_frames, -1, 1, features.shape[-1])  # [B, F, 512, 1, W]

        # ----- Temporal fusion (correctly handles 1 frame) -----
        if num_frames == 1:
            # Single frame: no fusion needed, just take the only frame
            fused = features[:, 0]                              # [B, 512, 1, W]
        elif self.fusion_fn is not None:
            # Use custom fusion (built from --fusion argument)
            feat_2d = features.squeeze(3)                       # [B, F, 512, W]
            fused_2d = self.fusion_fn(feat_2d)                  # [B, 512, W]
            fused = fused_2d.unsqueeze(2)                       # [B, 512, 1, W]
        else:
            # Default AttentionFusion (only for multi‑frame, expects total_frames = B * F)
            flat_features = features.view(B * num_frames, -1, 1, features.shape[-1])
            fused = self.default_fusion(flat_features)          # [B, 512, 1, W]

        # ----- Sequence modeling (BiLSTM + CTC head) -----
        seq_input = fused.squeeze(2).permute(0, 2, 1)           # [B, W, 512]
        rnn_out, _ = self.rnn(seq_input)                       # [B, W, 2*hidden]
        out = self.head(rnn_out)                               # [B, W, num_classes]
        logits = out.log_softmax(2)

        if return_frame_logits:
            return logits, None
        return logits