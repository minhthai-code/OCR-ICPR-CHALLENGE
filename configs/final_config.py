"""Configuration for final optimization (Stage B)."""
from dataclasses import dataclass, field
from typing import Dict, ClassVar
import torch


@dataclass
class Config:
    """Tunable configuration for maximising final performance.

    Most hyperparameters are set to sensible defaults but can be overridden
    via CLI or by modifying this file before running the experiment.
    """

    # Experiment tracking
    MODEL_TYPE: str = "restran_multiframe"   # best model type from ablation
    EXPERIMENT_NAME: str = "final_optimization"
    AUGMENTATION_LEVEL: str = "full"         # can be tuned: "full" or "light"
    BACKBONE: str = "proposed"               # best backbone from ablation
    PRETRAINED_BACKBONE: bool = False
    USE_STN: bool = True

    OPENOCR_ROOT: str = "third_party/OpenOCR"
    SVTRV2_REPO_URL: str = "https://github.com/Topdu/OpenOCR.git"
    SVTRV2_WEIGHTS_URL: str = "https://github.com/Topdu/OpenOCR/releases/download/develop0.0.1/openocr_svtrv2_ch.pth"

    DECODE_MODE: str = "reranked_beam"
    BEAM_WIDTH: int = 5
    LAYOUT_AWARE_DECODING: bool = True

    # Slot-based OCR
    NUM_SLOTS: int = 8
    CONSISTENCY_LOSS_WEIGHT: float = 0.02
    EMA_DECAY: float = 0.999
    DISTILL_WEIGHT: float = 0.05
    DISTILL_TEMPERATURE: float = 2.0
    SLOT_DIM: int = 512
    USE_EMA_TEACHER: bool = True

    # Data paths
    DATA_ROOT: str = "data/train"
    TEST_DATA_ROOT: str = "data/public_test"
    BLIND_DATA_ROOT: str = "data/blind_test"
    VAL_SPLIT_FILE: str = "data/val_tracks.json"
    SUBMISSION_FILE: str = "submission.txt"

    IMG_HEIGHT: int = 32
    IMG_WIDTH: int = 128

    # Character set
    CHARS: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # ─────────────────────────────────────────────────────────────────
    # TUNABLE HYPERPARAMETERS – adjust for best final performance
    # ─────────────────────────────────────────────────────────────────
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 8e-5
    EPOCHS: int = 45                # try 75 or 100
    SEED: int = 42                 # can be changed
    NUM_WORKERS: int = 4
    WEIGHT_DECAY: float = 1e-4      # try 5e-5, 1e-4
    GRAD_CLIP: float = 1.0
    SPLIT_RATIO: float = 0.9
    USE_CUDNN_BENCHMARK: bool = False

    # Learning rate scheduler
    USE_ONECYCLE_LR: bool = True
    DIV_FACTOR: float = 8.0
    FINAL_DIV_FACTOR: float = 20.0
    PCT_START: float = 0.2

    # Optional modules (enable best ones from ablation)
    USE_SLOT_EMBEDDING: bool = True
    SLOT_EMBED_DIM: int = 128
    USE_CONTRASTIVE_LOSS: bool = False
    CONTRASTIVE_WEIGHT: float = 0.05
    USE_CENTER_LOSS: bool = False
    CENTER_LOSS_WEIGHT: float = 0.01
    USE_HARD_CONFUSION_MINING: bool = True

    CONFUSION_GROUPS: ClassVar[list[list[str]]] = [
        ["O", "D", "Q"],
        ["R", "B"],
        ["1", "7"],
        ["2", "8"],
    ]

    # CRNN hyperparameters (not used if restran is chosen)
    HIDDEN_SIZE: int = 256
    RNN_DROPOUT: float = 0.3

    # ResTranOCR hyperparameters
    TRANSFORMER_HEADS: int = 8
    TRANSFORMER_LAYERS: int = 5
    TRANSFORMER_FF_DIM: int = 2048
    TRANSFORMER_DROPOUT: float = 0.1

    DEVICE: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    OUTPUT_DIR: str = "final_results"

    # Derived attributes
    CHAR2IDX: Dict[str, int] = field(default_factory=dict, init=False)
    IDX2CHAR: Dict[int, str] = field(default_factory=dict, init=False)
    NUM_CLASSES: int = field(default=0, init=False)

    def __post_init__(self):
        self.CHAR2IDX = {char: idx + 1 for idx, char in enumerate(self.CHARS)}
        self.IDX2CHAR = {idx + 1: char for idx, char in enumerate(self.CHARS)}
        self.NUM_CLASSES = len(self.CHARS) + 1