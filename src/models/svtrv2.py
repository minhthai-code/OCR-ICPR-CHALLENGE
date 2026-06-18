"""
Thin wrapper for the official OpenOCR SVTRv2 encoder.

The official implementation is used as is, without modifications.
This wrapper returns the native output of the encoder, which is a
3D tensor (B, T, C) with C = 256 (the internal feature dimension).
ResTran already supports 3D backbone outputs.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Locate and import the official SVTRv2 from third_party/OpenOCR
OPENOCR_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "OpenOCR"
if str(OPENOCR_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENOCR_ROOT))

from openrec.modeling.encoders.svtrv2 import SVTRv2 as OfficialSVTRv2


class SVTRv2Backbone(nn.Module):
    """
    Wrapper for the official OpenOCR SVTRv2.

    The official encoder returns a 3D tensor (B, T, C) with C = 256
    when last_stage=False. This matches the expected input format of
    ResTran when the backbone output is 3‑dimensional.

    Args:
        pretrained (bool): If True, load pretrained weights.
        pretrained_path (str, optional): Path to pretrained .pth file.
            If None and pretrained=True, uses default path under `weights/`.
        freeze_backbone (bool): Freeze all parameters.
    """
    def __init__(
        self,
        pretrained: bool = False,
        pretrained_path: str = None,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        # Instantiate the official encoder with last_stage=False
        # so that it returns (B, T, C) with C = 256 (internal dims)
        self.encoder = OfficialSVTRv2(
            max_sz=[32, 128],
            in_channels=3,
            out_channels=512,        # ignored because last_stage=False
            last_stage=False,        # Keep 3D token representation
        )

        # Load pretrained weights if requested
        if pretrained:
            self._load_pretrained(pretrained_path)
            print("SVTRv2: loading official OpenOCR pretrained weights")
        else:
            print("SVTRv2: random initialization (scratch)")

        # Optionally freeze all backbone parameters
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("SVTRv2: backbone frozen")

    def _load_pretrained(self, pretrained_path: str = None):
        """Load pretrained weights from OpenOCR checkpoint (partial load)."""
        if pretrained_path is None:
            # Default location: project_root/weights/openocr_svtrv2_ch.pth
            pretrained_path = (
                Path(__file__).resolve().parents[2] / "weights" / "openocr_svtrv2_ch.pth"
            )
        pretrained_path = Path(pretrained_path)

        if not pretrained_path.exists():
            raise FileNotFoundError(
                f"Pretrained checkpoint not found at {pretrained_path}.\n"
                f"Please download it using:\n"
                f"  curl -L -o weights/openocr_svtrv2_ch.pth https://github.com/Topdu/OpenOCR/releases/download/develop0.0.1/openocr_svtrv2_ch.pth"
            )

        print(f"Loading SVTRv2 pretrained weights from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Print first few keys to help debugging (optional)
        print(f"Checkpoint keys (first 10): {list(state_dict.keys())[:10]}")

        # Remove common prefixes that may have been added by DataParallel or wrapping
        def strip_prefix(key, prefixes):
            for p in prefixes:
                if key.startswith(p):
                    return key[len(p):]
            return key

        # Common prefixes in pretrained OpenOCR models
        prefixes_to_strip = ["module.", "model.", "backbone.", "encoder."]

        new_state_dict = {}
        for key, value in state_dict.items():
            # Try to map key to our model's expected key
            stripped_key = strip_prefix(key, prefixes_to_strip)
            # If still different, try to remove any remaining "encoder." inside
            if stripped_key.startswith("encoder."):
                stripped_key = stripped_key[8:]  # remove "encoder."
            new_state_dict[stripped_key] = value

        # Filter only keys that exist and have matching shapes in our encoder
        model_dict = self.encoder.state_dict()
        compatible = {}
        for key, value in new_state_dict.items():
            if key in model_dict and model_dict[key].shape == value.shape:
                compatible[key] = value
            else:
                # For debugging: print mismatches (optional)
                if key in model_dict and model_dict[key].shape != value.shape:
                    print(f"Shape mismatch for {key}: {value.shape} vs {model_dict[key].shape}")

        model_dict.update(compatible)
        self.encoder.load_state_dict(model_dict, strict=False)

        total_model_params = len(model_dict)
        loaded_params = len(compatible)
        print(f"Loaded {loaded_params} / {total_model_params} parameters "
              f"(missing: {total_model_params - loaded_params}).")

        if loaded_params == 0:
            raise RuntimeError(
                "No weights were loaded! Checkpoint key names do not match the model. "
                f"Checkpoint keys: {list(state_dict.keys())[:5]}. "
                f"Expected keys: {list(model_dict.keys())[:5]}. "
                "This may happen if the pretrained model uses a different architecture."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, 3, H, W) with H=32, W=128 typical.

        Returns:
            (B, T, C) with C = 256 – the native output of the official OpenOCR SVTRv2.
        """
        return self.encoder(x)