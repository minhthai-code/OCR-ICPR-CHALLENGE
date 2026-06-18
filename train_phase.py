"""Main entry point for pretraining OCR pipeline with weighted sampling, selective correction, and motion alignment."""

"""
    It orchestrates the entire pretraining pipeline, including:
        1. CLI arguments
            ↓
        2. Load Config & apply overrides
            ↓
        3. Build confusion groups (if selective correction enabled)
            ↓
        4. Create Dataset & WeightedRandomSampler
            ↓
        5. Create DataLoaders
            ↓
        6. Build Model (CRNN or ResTranOCR with optional refinement/motion alignment)
            ↓
        7. Resume from checkpoint (full resume support)
            ↓
        8. Train Model
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.dataset import MultiFrameDataset
from src.models.crnn import MultiFrameCRNN
from src.models.restran import ResTranOCR
from src.training.pretrain_trainer import PretrainTrainer as Trainer
from src.utils.common import seed_everything

# Helper to parse boolean flags (accepts "true"/"false", "True"/"False", "1"/"0")
def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "1", "yes", "y"):
        return True
    elif value.lower() in ("false", "0", "no", "n"):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got '{value}'")


# 1. CLI ARGUMENTS
def parse_args() -> argparse.Namespace:
    """Parse command line arguments for pretraining.

    Each argument has a default of None, it will fall back to the value
    defined in the Config class if not provided.
    """
    parser = argparse.ArgumentParser(
        description="Pretrain Multi-Frame OCR with Weighted Sampling, "
                    "Selective Correction, and Motion Alignment"
    )

    # Config selection (which hyperparameter set to use)
    parser.add_argument(
        "--config-type", type=str, default="ablation",
        choices=["ablation", "final"],
        help="Which configuration to use: ablation (fixed for Stage A) or final (tunable for Stage B)"
    )

    # Basic training overrides (can be used to quickly change hyperparameters)
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of training epochs (default: from config)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Batch size for training (default: from config)"
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Learning rate (default: from config)"
    )
    parser.add_argument(
        "--experiment-name", type=str, default=None,
        help="Experiment name for checkpoints (default: from config)"
    )
    parser.add_argument(
        "--model", type=str, choices=["crnn", "crnn_multiframe", "restran", "restran_multiframe"],
        default=None,
        help="Model architecture and frame mode (default: from config)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="Number of data loader workers (default: from config)"
    )
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Root directory for training data (default: from config)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="pretrain_results",
        help="Directory to save checkpoints (default: pretrain_results/)"
    )

    # Backbone specific (ResNet34, proposed lightweight SVTR, real SVTRv2)
    parser.add_argument(
        "--backbone", type=str, default=None,
        choices=["resnet34", "proposed", "svtrv2"],
        help="Backbone architecture: resnet34 (CNN), proposed (lightweight SVTR), svtrv2 (heavy reference)"
    )
    parser.add_argument(
        "--pretrained-backbone", type=str_to_bool, default=None,
        help="Load pretrained backbone weights (true/false). If not set, uses config default."
    )
    parser.add_argument(
        "--freeze-backbone", action="store_true",
        help="Freeze backbone parameters (not recommended for proposed or svtrv2)"
    )
    # NEW: Path to custom backbone weights (e.g., Kaggle dataset path)
    parser.add_argument(
        "--backbone-weights", type=str, default=None,
        help="Path to pretrained backbone weights file (overrides default location)"
    )

    # Temporal fusion strategy (only used for multi‑frame models)
    parser.add_argument(
        "--fusion", type=str, default="transformer",
        choices= ["mean", "max", "attention", "transformer", "reliability"],
        help="Temporal fusion strategy for multi‑frame models"
    )

    # Selective correction (refinement)
    parser.add_argument(
        "--enable-refinement", action="store_true",
        help="[DEPRECATED] Use --use_refinement true/false instead"
    )
    parser.add_argument(
        "--risk-loss-weight", type=float, default=0.02,
        help="Weight for risk loss (refinement)"
    )
    parser.add_argument(
        "--refine-loss-weight", type=float, default=0.5,
        help="Weight for refine loss (refinement)"
    )

    # Motion alignment – aligns consecutive frames to improve temporal consistency
    parser.add_argument(
        "--use_motion_alignment", type=str_to_bool, default=False,
        help="Enable motion alignment between frames (true/false)"
    )

    # STN – Spatial Transformer Network for rectification
    parser.add_argument(
        "--use_stn", type=str_to_bool, default=True,
        help="Use Spatial Transformer Network (true/false, default: true)"
    )

    # Refinement – selective correction
    parser.add_argument(
        "--use_refinement", type=str_to_bool, default=False,
        help="Enable selective correction (refinement head) (true/false)"
    )

    # Resume logic (supports full restart including optimizer state)
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--full-resume", action="store_true",
        help="Load model strictly (use when architecture unchanged)"
    )

    return parser.parse_args()


def build_confusion_groups(config) -> list:
    """Build confusion groups from character indices for selective correction.

    These groups define which characters are easily confused (e.g., O/D/Q).
    The refinement loss will try to correct mistakes within these groups.
    """
    raw_groups = [
        ("O", "D", "Q"),
        ("M", "N", "H"),
        ("6", "8", "4", "9"),
        ("2", "3"),
        ("V", "Y"),
        ("A", "B"),
        ("E", "C"),
        ("W", "V"),
        ("1", "7"),
    ]
    confusion_groups = []
    for group in raw_groups:
        indices = [config.CHAR2IDX[ch] for ch in group if ch in config.CHAR2IDX]
        if len(indices) >= 2:
            confusion_groups.append(indices)
    print(f" Built {len(confusion_groups)} confusion groups")
    return confusion_groups


def main() -> None:
    """Main pretraining entry point."""
    # Parse CLI arguments
    args = parse_args()

    # --model is required
    if args.model is None:
        raise ValueError("--model is required. Choose from: crnn, crnn_multiframe, restran, restran_multiframe")

    # Backward compatibility: if old --enable-refinement is used, map to --use_refinement
    if args.enable_refinement and not args.use_refinement:
        args.use_refinement = True

    # 2. Load Config (ablation or final) based on --config-type
    if args.config_type == "ablation":
        from configs.ablation_config import Config
    else:
        from configs.final_config import Config

    config = Config()
    config.EXPERIMENT_NAME = "pretrain_stage1"   # default name (can be overridden)

    # Mapping from CLI argument name to Config attribute name
    arg_to_config = {
        "epochs": "EPOCHS",
        "batch_size": "BATCH_SIZE",
        "lr": "LEARNING_RATE",
        "experiment_name": "EXPERIMENT_NAME",
        "model": "MODEL_TYPE",
        "num_workers": "NUM_WORKERS",
        "data_root": "DATA_ROOT",
        "output_dir": "OUTPUT_DIR",
        "backbone": "BACKBONE",
        "use_stn": "USE_STN",
        "pretrained_backbone": "PRETRAINED_BACKBONE",
    }

    """
        .items() in Python returns: key-value pairs of the dictionary as tuples.
        like ('epochs', 'EPOCHS'), ('batch_size', 'BATCH_SIZE'), etc.
        Python tuple unpacking in the loop: we assign first value (arg_name) and second value (config_name).
        getattr(args, arg_name, None): if the CLI argument was provided, it returns that value, else None.
        If the value is not None, we use setattr(config, config_name, value) to override the config.
        For example, if --epochs 50 is provided, it will set config.EPOCHS = 50.
    """
    for arg_name, config_name in arg_to_config.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            setattr(config, config_name, value)

    # --- Determine multi-frame vs single-frame and base model type ---
    is_multiframe = "multiframe" in args.model
    base_model_type = args.model.replace("_multiframe", "")
    fusion_name = args.fusion if is_multiframe else None

    # Backbone selection and pretrained flag handling
    if base_model_type == "restran":
        if config.BACKBONE == "proposed":
            # Proposed backbone has no pretrained version – enforce scratch
            config.PRETRAINED_BACKBONE = False
            print("\n✅ Proposed backbone selected (always trained from scratch)")
        elif config.BACKBONE == "svtrv2":
            print("\n✅ SVTRv2 backbone selected")
            # Pretrained flag will be taken from config (or CLI override)
        else:  # resnet34
            config.BACKBONE = "resnet34"
            print("\n✅ ResNet34 backbone selected")
            # Pretrained flag will be taken from config (or CLI override)
        print(f"    Pretrained = {config.PRETRAINED_BACKBONE}")
    else:  # crnn
        print(f"\n✅ CRNN uses fixed CNNBackbone")
        config.PRETRAINED_BACKBONE = True   # keep as is, but CRNN doesn't use it

    if args.freeze_backbone:
        print("⚠️  WARNING: --freeze-backbone set – not recommended for proposed/svtrv2!")

    # Setup output directory (where checkpoints and logs will be saved)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Seed everything for reproducibility
    """
    Purpose of seed_everything():
    This function ensures your machine learning experiments are reproducible,
    meaning you get the same results every time you run your code.
    It sets fixed random seeds for Python, NumPy, and PyTorch (CPU and GPU).
    This makes operations like weight initialization, data shuffling, and random sampling
    behave consistently across runs.

    GPU Behavior (cuDNN Settings):
      - Deterministic mode (benchmark=False) → slower but exact same results
      - Benchmark mode (benchmark=True) → faster but may produce slightly different results
    """
    seed_everything(config.SEED)

    # Print final configuration after CLI overrides
    print("\n" + "=" * 60)
    for key, value in vars(config).items():
        if not key.startswith("__"):
            print(f"  {key:30} : {value}")
    print("=" * 60 + "\n")

    # 3. Build confusion groups if refinement enabled
    use_refinement = args.use_refinement
    confusion_groups = None
    if use_refinement and config.MODEL_TYPE in ("restran", "restran_multiframe"):
        confusion_groups = build_confusion_groups(config)
        print(f"\n✨ Selective Correction ENABLED")
        print(f"   Risk Loss Weight: {args.risk_loss_weight}")
        print(f"   Refine Loss Weight: {args.refine_loss_weight}")
    elif use_refinement and config.MODEL_TYPE not in ("restran", "restran_multiframe"):
        print("⚠️  Selective correction only available for restran models – disabling.")
        use_refinement = False

    # Common dataset parameters (shared by train and val datasets)
    common_ds_params = {
        "split_ratio": config.SPLIT_RATIO,
        "img_height": config.IMG_HEIGHT,
        "img_width": config.IMG_WIDTH,
        "char2idx": config.CHAR2IDX,
        "val_split_file": config.VAL_SPLIT_FILE,
        "seed": config.SEED,
        "augmentation_level": config.AUGMENTATION_LEVEL,
    }

    # 4. Create Dataset and WeightedRandomSampler
    """
    MultiFrameDataset loads image sequences and their labels.
    - mode='train': applies data augmentation, returns (images, labels, frame_counts)
    - mode='val':   no augmentation, used for validation
    The dataset automatically splits data into train/val based on split_ratio or a saved split file.
    """
    train_ds = MultiFrameDataset(config.DATA_ROOT, mode="train", **common_ds_params)
    val_ds = MultiFrameDataset(config.DATA_ROOT, mode="val", **common_ds_params)

    # WeightedRandomSampler for class balancing (handles imbalanced datasets)
    # sample_weights are computed by the dataset (inversely proportional to class frequency)
    train_sampler = WeightedRandomSampler(
        weights=train_ds.sample_weights,
        num_samples=len(train_ds),
        replacement=True
    )

    # 5. Create DataLoaders
    """
    DataLoader converts dataset into batches for efficient GPU processing.
    - train_loader uses sampler (no shuffle argument) and weighted sampling.
    - val_loader uses shuffle=False because evaluation order doesn't matter.
    collate_fn handles variable-length sequences (different number of frames per sample).
    pin_memory speeds up data transfer to GPU.
    """
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        sampler=train_sampler,               # replaces shuffle=True
        num_workers=config.NUM_WORKERS,
        collate_fn=MultiFrameDataset.collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=MultiFrameDataset.collate_fn,
        pin_memory=True,
    )
    print(f"\n Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    # 6. Build Model
    """
    Two model families:
      - restran: Transformer-based OCR (supports selective correction, motion alignment, and fusion)
      - crnn:    CNN + RNN baseline (simpler, no refinement)
    The model is moved to the device specified in config (GPU if available, else CPU).

    Multi‑frame vs single‑frame is determined by the presence of "_multiframe" in the model name.
    Fusion strategy is only used when multi‑frame is True.
    """
    # is_multiframe and base_model_type already computed earlier
    # fusion_name already computed

    if base_model_type == "restran":
        model = ResTranOCR(
            num_classes=config.NUM_CLASSES,
            num_slots=config.NUM_SLOTS,
            cnn_channels=config.SLOT_DIM,
            transformer_heads=config.TRANSFORMER_HEADS,
            transformer_layers=config.TRANSFORMER_LAYERS,
            transformer_ff_dim=config.TRANSFORMER_FF_DIM,
            dropout=config.TRANSFORMER_DROPOUT,
            use_stn=args.use_stn,
            pretrained_backbone=config.PRETRAINED_BACKBONE,
            use_refinement=use_refinement,
            confusion_groups=confusion_groups,
            backbone_name=config.BACKBONE,
            openocr_root=config.OPENOCR_ROOT,
            svtrv2_weights_url=config.SVTRV2_WEIGHTS_URL,
            freeze_backbone=args.freeze_backbone,
            use_motion_alignment=args.use_motion_alignment,
            is_multiframe=is_multiframe,
            fusion_name=fusion_name,
            backbone_weights_path=args.backbone_weights,   # NEW: pass custom path
        ).to(config.DEVICE)
    else:   # crnn
        if use_refinement:
            print("⚠️  Refinement only for restran model – disabling.")
            use_refinement = False
        model = MultiFrameCRNN(
            num_classes=config.NUM_CLASSES,
            hidden_size=config.HIDDEN_SIZE,
            use_stn=args.use_stn,
            is_multiframe=is_multiframe,
            fusion_name=fusion_name,
        ).to(config.DEVICE)

    print(f"\n✅ Model ready: {args.model} | Backbone: {args.backbone} | Fusion: {args.fusion}")

    # Pass loss weights for selective correction to config (used by trainer)
    config.RISK_LOSS_WEIGHT = args.risk_loss_weight
    config.REFINE_LOSS_WEIGHT = args.refine_loss_weight

    # Initialize trainer
    """
    Trainer handles:
      - training loop
      - forward pass
      - loss computation (CTC + optional risk/refine losses)
      - backpropagation & optimizer step
      - validation after each epoch
      - checkpoint saving (best model and latest)
      - learning rate scheduling
    """
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        idx2char=config.IDX2CHAR,
    )

    # 7. Resume logic (full resume support)
    """
    If --resume /path/to/checkpoint.pth is provided:
      - Load model state dict (and optimizer state if full_resume is True)
      - Continue training from the saved epoch + 1
    strict=not args.full_resume means:
      - If full_resume=True → strict loading (exact architecture match)
      - If full_resume=False → allow missing/unexpected keys (useful when model changed slightly)
    """
    start_epoch = 0
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"Checkpoint not found: {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume, strict=not args.full_resume)
        print(f"✅ Resuming from epoch {start_epoch + 1}")

    # 8. Train Model
    print("\n🚀 Starting pretraining...\n")
    trainer.fit(start_epoch=start_epoch)

    print(f"\n🎉 Done. Saved to: {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()