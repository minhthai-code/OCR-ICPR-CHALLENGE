"""Reusable model components for multi-frame OCR."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet34_Weights, resnet34


class FrameAlignment(nn.Module):
    """Learn to align all frames to the first frame via affine transformations.

    Uses a shared CNN to extract features from all frames in one pass,
    then predicts affine parameters from (reference, current, abs_diff) pairs.
    Fully batched – no Python loops over frames.
    Identity initialization ensures stable training start.
    """
    def __init__(self, in_channels=3, hidden_dim=64):
        super().__init__()
        # Shared feature extractor for all frames
        self.feature_net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 8)),   # [B, hidden_dim, 4, 8]
        )
        # Affine head: input = concat(reference, current, abs_diff)
        # reference + current + abs_diff = 3 * hidden_dim * 4 * 8
        in_features = 3 * hidden_dim * 4 * 8
        self.affine_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 6)   # 6 affine params: a11, a12, a13, a21, a22, a23
        )
        self._init_affine_as_identity()

    def _init_affine_as_identity(self):
        """Initialize last linear layer to output identity transformation."""
        with torch.no_grad():
            # weights = 0, bias = [1,0,0,0,1,0]
            self.affine_head[-1].weight.zero_()
            self.affine_head[-1].bias.copy_(
                torch.tensor([1., 0., 0., 0., 1., 0.])
            )

    def forward(self, x):
        """
        x: [B, F, C, H, W]  - F frames per batch
        returns: aligned frames [B, F, C, H, W] (first frame unchanged)
        """
        B, num_frames, C, H, W = x.shape
        # ---- 1. Extract features for all frames in one batch ----
        x_flat = x.reshape(B * num_frames, C, H, W)      # [B*F, C, H, W]
        feats = self.feature_net(x_flat)                 # [B*F, D, 4, 8]
        feats = feats.reshape(B, num_frames, -1)         # [B, F, D*4*8]

        # ---- 2. Build pairs (reference is frame 0) ----
        ref_feat = feats[:, 0:1, :]                      # [B, 1, L]
        cur_feat = feats[:, 1:, :]                       # [B, F-1, L]

        # Use absolute difference for motion cue (more stable)
        abs_diff = torch.abs(ref_feat - cur_feat)        # [B, F-1, L]

        # Concatenate reference (broadcasted), current, and absolute difference
        pair_feat = torch.cat([
            ref_feat.expand_as(cur_feat),                # [B, F-1, L]
            cur_feat,                                    # [B, F-1, L]
            abs_diff                                     # [B, F-1, L]
        ], dim=-1)                                       # [B, F-1, 3*L]

        # ---- 3. Predict affine parameters for frames 1..F-1 ----
        theta = self.affine_head(pair_feat)              # [B, F-1, 6]
        theta = theta.reshape(B, -1, 2, 3)               # [B, F-1, 2, 3]

        # ---- 4. Vectorized warping of all non‑reference frames ----
        # Get all frames except the reference
        non_ref_frames = x[:, 1:, ...]                   # [B, F-1, C, H, W]
        # Flatten batch and frames for batch grid_sample
        B_nr = B * (num_frames - 1)
        non_ref_flat = non_ref_frames.reshape(B_nr, C, H, W)
        theta_flat = theta.reshape(B_nr, 2, 3)

        # Generate grid and warp all in one go
        grid = F.affine_grid(theta_flat, non_ref_flat.size(), align_corners=False)
        warped_flat = F.grid_sample(non_ref_flat, grid, align_corners=False)

        # Reshape back to [B, F-1, C, H, W]
        warped = warped_flat.reshape(B, num_frames - 1, C, H, W)

        # ---- 5. Prepend the reference frame (unchanged) ----
        reference = x[:, 0:1, ...]                       # [B, 1, C, H, W]
        aligned = torch.cat([reference, warped], dim=1)  # [B, F, C, H, W]

        return aligned

class RiskGate(nn.Module):
    """
    Predicts how risky each slot is.
    Uses slot features + simple uncertainty statistics.
    Output: [B, num_slots, 1] in [0, 1]
    """
    def __init__(
        self,
        slot_dim: int,
        num_classes: int,
        hidden_dim: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = hidden_dim or max(64, slot_dim // 2)

        in_dim = slot_dim + 3  # slot feature + entropy + max_prob + margin
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, slot_features: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slot_features: [B, S, D]
            probs: [B, S, C] softmax probabilities
        """
        eps = 1e-8
        log_probs = (probs + eps).log()

        entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)
        entropy = entropy / math.log(probs.size(-1))  # normalize to roughly [0, 1]

        top2 = probs.topk(k=2, dim=-1).values
        max_prob = top2[..., :1]
        margin = top2[..., :1] - top2[..., 1:2]

        x = torch.cat([slot_features, entropy, max_prob, margin], dim=-1)
        return self.net(x)        # raw logits


class ConfusionRefiner(nn.Module):
    """
    Predicts a residual correction over classes.
    The caller supplies a per-slot confusion mask so the delta only touches
    the active confusion group.
    """
    def __init__(
        self,
        slot_dim: int,
        num_classes: int,
        hidden_dim: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = hidden_dim or max(128, slot_dim)

        in_dim = slot_dim + num_classes + 1  # slot feature + probs + risk
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # Start from near-identity behavior.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        slot_features: torch.Tensor,
        probs: torch.Tensor,
        risk: torch.Tensor,
        confusion_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            slot_features: [B, S, D]
            probs: [B, S, C]
            risk: [B, S, 1]
            confusion_mask: [B, S, C] soft weights (0..1) for each class
        Returns:
            delta_logits: [B, S, C]
        """
        x = torch.cat([slot_features, probs, risk], dim=-1)
        delta = self.net(x)
        return delta * confusion_mask


class TemporalFusionTransformer(nn.Module):
    """
    Cross-frame transformer for fusing multi-frame token features.
    Input:  [B, F, T, D]
    Output: [B, F*T, D]
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
            [B, F*T, D]
        """
        B, n_frames, T, D = x.shape
        x_flat = x.view(B, n_frames * T, D)

        # add frame identity so the transformer knows which token came from which frame
        frame_ids = torch.arange(n_frames, device=x.device).unsqueeze(1).repeat(1, T).reshape(1, n_frames * T)
        frame_ids = frame_ids.expand(B, -1)  # [B, F*T]
        x_flat = x_flat + self.frame_embedding(frame_ids)

        x_flat = self.pos_encoding(x_flat)
        fused = self.transformer(x_flat)
        return fused


class SlotAttentionPlateHead(nn.Module):
    """
    Slot-based character head for license plates.

    Returns:
    - logits: [B, num_slots, num_classes]
    - frame_logits: [B, F, num_slots, num_classes] if requested
    - slot_features: [B, num_slots, embed_dim] if requested

    Optional selective refinement:
    - risk gate
    - confusion-group residual correction
    """
    def __init__(
        self,
        num_slots: int = 7,
        slot_dim: int = 512,
        num_classes: int = 37,
        n_iter: int = 3,
        embed_dim: int = 192,
        refine: bool = True,
        confusion_groups: list = None,   # list of lists of class indices
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_classes = num_classes
        self.n_iter = n_iter
        self.embed_dim = embed_dim
        self.refine = refine

        self.slot_embed = nn.Embedding(num_slots, slot_dim)
        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_dim))

        self.to_q = nn.Linear(slot_dim, slot_dim)
        self.to_k = nn.Linear(slot_dim, slot_dim)
        self.to_v = nn.Linear(slot_dim, slot_dim)

        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim),
            nn.ReLU(inplace=True),
            nn.Linear(slot_dim, slot_dim),
        )

        self.to_logits = nn.Linear(slot_dim, num_classes)

        self.to_embed = nn.Sequential(
            nn.Linear(slot_dim, slot_dim),
            nn.ReLU(inplace=True),
            nn.Linear(slot_dim, embed_dim),
        )

        # Selective correction modules.
        self.risk_gate = RiskGate(
            slot_dim=embed_dim,
            num_classes=num_classes,
            hidden_dim=max(64, embed_dim * 2),
            dropout=0.1,
        )

        self.confusion_refiner = ConfusionRefiner(
            slot_dim=embed_dim,
            num_classes=num_classes,
            hidden_dim=max(128, embed_dim * 2),
            dropout=0.1,
        )

        # Build confusion masks.
        confusion_groups = confusion_groups or []
        masks = []
        for group in confusion_groups:
            mask = torch.zeros(num_classes, dtype=torch.float32)
            for idx in group:
                if 0 <= int(idx) < num_classes:
                    mask[int(idx)] = 1.0
            if mask.sum() >= 2:
                masks.append(mask)

        if len(masks) == 0:
            self.register_buffer("confusion_group_matrix", torch.zeros(1, num_classes))
            self.has_confusion_groups = False
        else:
            self.register_buffer("confusion_group_matrix", torch.stack(masks, dim=0))
            self.has_confusion_groups = True

    def _init_slots(self, batch_size: int, device: torch.device):
        mu = self.slots_mu.expand(batch_size, self.num_slots, -1)
        sigma = self.slots_log_sigma.exp().expand(batch_size, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)

        pos_ids = torch.arange(self.num_slots, device=device)
        slots = slots + self.slot_embed(pos_ids).unsqueeze(0)
        return slots

    def _refine_slots(self, tokens: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        batch_size = tokens.shape[0]
        D = self.slot_dim
        for _ in range(self.n_iter):
            slots_prev = slots
            q = self.to_q(slots)
            k = self.to_k(tokens)
            v = self.to_v(tokens)

            attn = torch.einsum("bsd,bnd->bsn", q, k) / math.sqrt(D)
            attn = F.softmax(attn, dim=-1)

            attn = attn + 1e-8
            attn_norm = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum("bsn,bnd->bsd", attn_norm, v)

            slots = self.gru(
                updates.reshape(-1, D),
                slots_prev.reshape(-1, D),
            ).view(batch_size, self.num_slots, D)

            slots = slots + self.mlp(slots)
        return slots

    def forward(
        self,
        x,
        return_frame_logits: bool = False,
        frame_tokens=None,
        return_slot_features: bool = False,
        use_refinement: bool = False,
        return_both_logits: bool = False,
        return_aux: bool = False,
    ):
        """
        If return_aux=True, returns a dict with:
            logits, base_logits, refined_logits, risk_scores, chosen_group_idx
        """
        if x.dim() == 4:
            B, n_f, T, D = x.shape
            tokens = x.view(B, n_f * T, D)
            if frame_tokens is None:
                frame_tokens = x
        elif x.dim() == 3:
            B, N, D = x.shape
            tokens = x
        else:
            raise ValueError(f"Expected x with 3 or 4 dims, got {x.shape}")

        initial_slots = self._init_slots(B, x.device)
        slots = self._refine_slots(tokens, initial_slots)

        base_logits = self.to_logits(slots)      # [B, S, C]
        slot_features = self.to_embed(slots)     # [B, S, E]

        probs = F.softmax(base_logits, dim=-1)

        # Risk gate (raw logits)
        risk_scores = self.risk_gate(slot_features, probs)  # [B, S, 1]

        # ----- SOFT CONFUSION ROUTING (replaces hard argmax) -----
        if self.has_confusion_groups:
            # group_mass: [B, S, num_groups]
            group_mass = torch.einsum("bsc,gc->bsg", probs, self.confusion_group_matrix)
            # temperature 0.5 makes softmax sharper but still differentiable
            group_weights = F.softmax(group_mass / 0.5, dim=-1)
            # confusion_mask: soft mixture over classes from all groups
            confusion_mask = torch.einsum("bsg,gc->bsc", group_weights, self.confusion_group_matrix)
        else:
            group_mass = None
            confusion_mask = torch.ones_like(base_logits)

        refined_logits = None
        delta_logits = None
        if self.refine and (use_refinement or return_both_logits or return_aux):
            delta_logits = self.confusion_refiner(
                slot_features=slot_features,
                probs=probs,
                risk=risk_scores,
                confusion_mask=confusion_mask,
            )
            # ----- REFINEMENT FLOOR (never turns off completely) -----
            risk_probs = torch.sigmoid(risk_scores)
            risk_scale = 0.25 + 0.75 * risk_probs   # always at least 0.25
            refined_logits = base_logits + risk_scale * delta_logits

        if return_both_logits and self.refine:
            logits_out = (base_logits, refined_logits)
        elif use_refinement and self.refine and refined_logits is not None:
            logits_out = refined_logits
        else:
            logits_out = base_logits

        frame_logits = None
        if return_frame_logits and frame_tokens is not None:
            Bf, num_frames, Tf, Df = frame_tokens.shape
            per_frame = frame_tokens.reshape(Bf * num_frames, Tf, Df)
            f_slots_init = self._init_slots(Bf * num_frames, x.device)
            f_slots = self._refine_slots(per_frame, f_slots_init)
            frame_logits = self.to_logits(f_slots).view(Bf, num_frames, self.num_slots, -1)

        if return_aux:
            return {
                "logits": logits_out,
                "base_logits": base_logits,
                "refined_logits": refined_logits,
                "risk_scores": risk_scores,
                "delta_logits": delta_logits,
                "chosen_group_idx": None,      # no longer used, kept for compatibility
                "group_mass": group_mass,
                "frame_logits": frame_logits,
                "slot_features": slot_features,
            }

        if return_frame_logits and return_slot_features:
            return logits_out, frame_logits, slot_features
        if return_frame_logits:
            return logits_out, frame_logits
        if return_slot_features:
            return logits_out, slot_features
        return logits_out


class STNBlock(nn.Module):
    """Spatial Transformer Network (STN) for image alignment."""
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.MaxPool2d(2, 2), nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((4, 8)) 
        )
        self.fc_loc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 8, 128), nn.ReLU(True),
            nn.Linear(128, 6)
        )
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = self.localization(x)
        theta = self.fc_loc(xs).view(-1, 2, 3)
        return theta


class AttentionFusion(nn.Module):
    """Attention-based fusion module for combining multi-frame features."""
    def __init__(self, channels: int):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total_frames, c, h, w = x.size()
        num_frames = 5
        batch_size = total_frames // num_frames
        x_view = x.view(batch_size, num_frames, c, h, w)
        scores = self.score_net(x).view(batch_size, num_frames, 1, h, w)
        weights = F.softmax(scores, dim=1)
        return torch.sum(x_view * weights, dim=1)


class CNNBackbone(nn.Module):
    """A simple CNN backbone for CRNN baseline (Unused by ResTran but kept for reference)."""
    def __init__(self, out_channels=512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(512, out_channels, 2, 1, 0), nn.BatchNorm2d(out_channels), nn.ReLU(True)
        )
    def forward(self, x):
        return self.features(x)


class ResNetFeatureExtractor(nn.Module):
    """ResNet-based backbone customized for OCR."""
    def __init__(self, pretrained: bool = False, use_multiscale: bool = True):
        super().__init__()
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        resnet = resnet34(weights=weights)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Make ResNet better for OCR width preservation
        self.layer3[0].conv1.stride = (2, 1)
        self.layer3[0].downsample[0].stride = (2, 1)
        self.layer4[0].conv1.stride = (2, 1)
        self.layer4[0].downsample[0].stride = (2, 1)

        self.use_multiscale = use_multiscale
        if use_multiscale:
            self.ms2 = MultiScaleResidualBlock(128, dropout=0.0)
            self.ms3 = BottleneckMultiScaleResidualBlock(256, hidden_ratio=0.5, dropout=0.0)
        else:
            self.ms2 = nn.Identity()
            self.ms3 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.ms2(x)

        x = self.layer3(x)
        x = self.ms3(x)

        x = self.layer4(x)

        # Better than plain mean for OCR stroke preservation
        x = F.adaptive_max_pool2d(x, (1, x.shape[-1]))
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding."""
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


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class MultiScaleResidualBlock(nn.Module):
    """
    Lightweight parallel multi-scale block inspired by the paper idea.
    Uses 3x3 / 5x5 / 7x7 branches, then fuses them back with a residual connection.
    """
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        c3 = channels // 3
        c5 = channels // 3
        c7 = channels - c3 - c5

        self.branch3 = ConvBNReLU(channels, c3, kernel_size=3, stride=1, padding=1)
        self.branch5 = ConvBNReLU(channels, c5, kernel_size=5, stride=1, padding=2)
        self.branch7 = ConvBNReLU(channels, c7, kernel_size=7, stride=1, padding=3)

        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = torch.cat(
            [self.branch3(x), self.branch5(x), self.branch7(x)],
            dim=1
        )
        y = self.fuse(y)
        y = self.dropout(y)
        return self.act(x + y)


class BottleneckMultiScaleResidualBlock(nn.Module):
    """
    Stronger multiscale block:
    1x1 reduce -> parallel 3x3 / 5x5 / 7x7 -> 1x1 fuse -> residual
    """
    def __init__(self, channels: int, hidden_ratio: float = 0.5, dropout: float = 0.0):
        super().__init__()
        hidden = max(32, int(channels * hidden_ratio))

        self.reduce = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )

        c3 = hidden // 3
        c5 = hidden // 3
        c7 = hidden - c3 - c5

        self.branch3 = nn.Sequential(
            nn.Conv2d(hidden, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(hidden, c5, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(c5),
            nn.ReLU(inplace=True),
        )
        self.branch7 = nn.Sequential(
            nn.Conv2d(hidden, c7, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(c7),
            nn.ReLU(inplace=True),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        z = self.reduce(x)
        y = torch.cat([self.branch3(z), self.branch5(z), self.branch7(z)], dim=1)
        y = self.fuse(y)
        y = self.dropout(y)
        return self.act(x + y)