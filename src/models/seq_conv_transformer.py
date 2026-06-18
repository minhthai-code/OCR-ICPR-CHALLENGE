import torch
import torch.nn as nn
import torch.nn.functional as F


# LIGHTWEIGHT SEQUENCE-ORIENTED CONV-TRANSFORMER ENCODER
class SeqConvTransformerEncoder(nn.Module):
    def __init__(
        self,
        pretrained=False,   # ignored (we use clean training)
        out_channels=512,
        freeze_backbone=False,
        img_size=(32, 128),
    ):
        super().__init__()

        self.out_channels = out_channels

        # Stage 1
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),  # ↓H/2
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Stage 2
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # ↓H/4
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Stage 3 global modeling
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=(2,1), padding=1),  # ↓H only
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.project = nn.Linear(256, out_channels)

        if freeze_backbone:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        # [B, C, H, W] → collapse height
        x = F.adaptive_avg_pool2d(x, (1, x.shape[-1]))
        x = x.squeeze(2)             # [B, C, W]
        x = x.permute(0, 2, 1)       # [B, T, C]

        x = self.transformer(x)
        x = self.project(x)

        return x