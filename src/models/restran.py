"""ResTranOCR: Advanced OCR for multi-frame license plates with STN + ResNet/SVTRv2 + Temporal Fusion + SlotAttention."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.components import (
    ResNetFeatureExtractor,
    SlotAttentionPlateHead,
    STNBlock,
    FrameAlignment
)
from src.models.seq_conv_transformer import SeqConvTransformerEncoder
from src.fusion.fusion import build_fusion
from src.models.components import TemporalFusionTransformer as OldTemporalFusion


class ResTranOCR(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_slots: int = 7,
        cnn_channels: int = 512,
        transformer_heads: int = 8,
        transformer_layers: int = 2,
        transformer_ff_dim: int = 2048,
        dropout: float = 0.1,
        use_stn: bool = True,
        pretrained_backbone: bool = False,
        use_multiscale: bool = True,
        embed_dim: int = 128,
        use_refinement: bool = False,
        confusion_groups: list = None,
        backbone_name: str = "resnet34",
        openocr_root: str = "third_party/OpenOCR",
        svtrv2_weights_url: str = None,
        freeze_backbone: bool = False,
        use_motion_alignment: bool = False,
        is_multiframe: bool = True,
        fusion_name: str = "transformer",
        backbone_weights_path: str = None,
    ):
        super().__init__()
        self.cnn_channels = cnn_channels
        self.num_slots = num_slots
        self.use_stn = use_stn
        self.use_refinement = use_refinement
        self.backbone_name = backbone_name.lower()
        self.freeze_backbone = freeze_backbone
        self.use_motion_alignment = use_motion_alignment
        self.is_multiframe = is_multiframe
        self.fusion_name = fusion_name

        print(f" Using backbone: {self.backbone_name}")
        if use_motion_alignment:
            print(" Motion alignment ENABLED")
        if not is_multiframe:
            print("ResTran: SINGLE-FRAME mode (using only first frame)")
        else:
            print(f"ResTran: MULTI-FRAME mode (5 frames) with fusion: {fusion_name}")

        if self.use_stn:
            self.stn = STNBlock(in_channels=3)

        # Backbone selection
        if self.backbone_name == "proposed":
            self.backbone = SeqConvTransformerEncoder(
                pretrained=pretrained_backbone,
                out_channels=cnn_channels,
                freeze_backbone=self.freeze_backbone,
                img_size=(32, 128),
            )
            self.backbone_proj = None
            print(f"Proposed backbone: output channels = {cnn_channels}")

        elif self.backbone_name == "svtrv2":
            from src.models.svtrv2 import SVTRv2Backbone
            self.backbone = SVTRv2Backbone(
                pretrained=pretrained_backbone,
                pretrained_path=backbone_weights_path,
                freeze_backbone=self.freeze_backbone,
            )
            # Official SVTRv2 outputs 256 channels.
            # Project to cnn_channels (512) if needed.
            if cnn_channels != 256:
                self.backbone_proj = nn.Linear(256, cnn_channels)
                print(f"SVTRv2: native output dim 256 → projected to {cnn_channels}")
            else:
                self.backbone_proj = None
                print(f"SVTRv2: native output dim 256 (no projection)")

        elif self.backbone_name == "resnet34":
            self.backbone = ResNetFeatureExtractor(
                pretrained=pretrained_backbone,
                use_multiscale=use_multiscale
            )
            self.backbone_proj = None
            print(f"ResNet34 backbone: output channels = {cnn_channels}")

        else:
            raise ValueError(f"Unknown backbone_name: {self.backbone_name}")

        # Motion alignment module (only if enabled)
        if use_motion_alignment:
            self.alignment = FrameAlignment(in_channels=3)

        # Fusion module (built from name) – only if multi-frame
        if self.is_multiframe:
            if fusion_name == "transformer":
                # Use OLD fusion from components.py
                print("using Temporal fusion from components")
                self.fusion = OldTemporalFusion(
                    d_model=cnn_channels,
                    nhead=transformer_heads,
                    num_layers=transformer_layers,
                    dim_feedforward=transformer_ff_dim,
                    dropout=dropout,
                )
            else:
                self.fusion = build_fusion(
                    name=fusion_name,
                    d_model=cnn_channels,
                    nhead=transformer_heads,
                    num_layers=transformer_layers,
                    dim_feedforward=transformer_ff_dim,
                    dropout=dropout,
                )
        else:
            self.fusion = None

        self.head = SlotAttentionPlateHead(
            num_slots=self.num_slots,
            slot_dim=self.cnn_channels,
            num_classes=num_classes,
            n_iter=3,
            embed_dim=embed_dim,
            refine=use_refinement,
            confusion_groups=confusion_groups,
        )

    def forward(self, x, return_frame_logits=False, return_slot_features=False,
                use_refined_logits=False, return_both_logits=False, return_aux=False):
        B, num_frames, C, H, W = x.shape

        if not self.is_multiframe:
            # Single‑frame: take only the first frame
            x = x[:, 0:1, ...]
            num_frames = 1

        if self.use_motion_alignment:
            x = self.alignment(x)

        x_flat = x.reshape(B * num_frames, C, H, W)

        if self.use_stn:
            theta = self.stn(x_flat)
            grid = F.affine_grid(theta, x_flat.size(), align_corners=False)
            x_aligned = F.grid_sample(x_flat, grid, align_corners=False, padding_mode="border")
        else:
            x_aligned = x_flat

        backbone_out = self.backbone(x_aligned)

        # Optional projection (e.g., 256 → 512 for SVTRv2)
        if hasattr(self, 'backbone_proj') and self.backbone_proj is not None:
            backbone_out = self.backbone_proj(backbone_out)

        if backbone_out.dim() == 4:
            _, c, h, w = backbone_out.shape
            assert h == 1, f"Expected backbone height 1, got {h}"
            frame_tokens = backbone_out.view(B, num_frames, c, w)
            frame_tokens = frame_tokens.permute(0, 1, 3, 2).contiguous()
        elif backbone_out.dim() == 3:
            BF, t, c = backbone_out.shape
            assert BF == B * num_frames, f"Mismatch: {BF} vs {B*num_frames}"
            frame_tokens = backbone_out.reshape(B, num_frames, t, c)
        else:
            raise ValueError(f"Unexpected backbone output shape: {tuple(backbone_out.shape)}")

        # Apply fusion if multi‑frame, else take first frame's tokens
        if self.is_multiframe:
            fused_tokens = self.fusion(frame_tokens)   # expects (B, F, T, D) -> (B, T, D)
        else:
            # Single‑frame: remove the frame dimension
            fused_tokens = frame_tokens[:, 0]           # [B, T, D]

        if return_aux:
            return self.head(
                fused_tokens,
                return_frame_logits=return_frame_logits,
                frame_tokens=frame_tokens,
                return_slot_features=return_slot_features,
                use_refinement=(use_refined_logits or return_both_logits) and self.use_refinement,
                return_both_logits=return_both_logits and self.use_refinement,
                return_aux=True,
            )

        head_out = self.head(
            fused_tokens,
            return_frame_logits=return_frame_logits,
            frame_tokens=frame_tokens,
            return_slot_features=return_slot_features,
            use_refinement=(use_refined_logits or return_both_logits) and self.use_refinement,
            return_both_logits=return_both_logits and self.use_refinement,
            return_aux=False,
        )

        if return_frame_logits and return_slot_features:
            logits_part, frame_logits, slot_features = head_out
            return logits_part, frame_logits, slot_features
        if return_frame_logits:
            logits_part, frame_logits = head_out
            return logits_part, frame_logits
        if return_slot_features:
            logits_part, slot_features = head_out
            return logits_part, slot_features
        return head_out