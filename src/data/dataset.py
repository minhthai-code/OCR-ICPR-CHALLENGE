"""MultiFrameDataset for license plate recognition with multi-frame input."""
import glob
import json
import os
import random
from typing import Any, Dict, List, Tuple

import cv2
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.data.transforms import (
    get_train_transforms,
    get_val_transforms,
    get_degradation_transforms,
    get_light_transforms,
)

from torch.nn.utils.rnn import pad_sequence


class MultiFrameDataset(Dataset):
    """Dataset for multi-frame license plate recognition.
    
    Handles both real LR images and synthetic LR (degraded HR) images.
    Implements Scenario-B specific validation splitting logic.
    """
    
    def __init__(
        self,
        root_dir: str,
        mode: str = 'train',
        split_ratio: float = 0.9,
        img_height: int = 32,
        img_width: int = 128,
        char2idx: Dict[str, int] = None,
        val_split_file: str = "data/val_tracks.json",
        seed: int = 42,
        augmentation_level: str = "full",
        is_test: bool = False,
        full_train: bool = False,
    ):
        self.mode = mode
        self.samples: List[Dict[str, Any]] = []
        self.sample_weights: List[float] = []       
        self.img_height = img_height
        self.img_width = img_width
        self.char2idx = char2idx or {}
        self.val_split_file = val_split_file
        self.seed = seed
        self.augmentation_level = augmentation_level
        self.is_test = is_test
        self.full_train = full_train

        if mode == 'train':
            if augmentation_level == "light":
                self.transform = get_light_transforms(img_height, img_width)
            else:
                self.transform = get_train_transforms(img_height, img_width)
            self.degrade = get_degradation_transforms()
        else:
            self.transform = get_val_transforms(img_height, img_width)
            self.degrade = None

        print(f"[{mode.upper()}] Scanning: {root_dir}")

        abs_root = os.path.abspath(root_dir)
        search_path = os.path.join(abs_root, "**", "track_*")
        all_tracks = sorted(glob.glob(search_path, recursive=True))

        if not all_tracks:
            print(f"ERROR: No data found in '{root_dir}' with pattern '{search_path}'. Please check your data path.")
            return

        if is_test:
            print(f"[TEST] Loaded {len(all_tracks)} tracks.")
            self._index_test_samples(all_tracks)
            print(f"TOTAL: {len(self.samples)} test samples.")
        else:
            train_tracks, val_tracks = self._load_or_create_split(all_tracks, split_ratio)
            selected_tracks = train_tracks if mode == 'train' else val_tracks
            print(f"[{mode.upper()}] Loaded {len(selected_tracks)} tracks.")
            self._index_samples(selected_tracks)
            print(f"-> TOTAL: {len(self.samples)} samples.")

    # --------------------------------------------------------------------------
    # Weight computation helper
    # --------------------------------------------------------------------------
    HARD_CHAR_WEIGHTS = {
        "O": 1.3, "D": 1.3, "Q": 1.3,
        "M": 1.4, "H": 1.4,
        "6": 1.2, "8": 1.2, "4": 1.15,
        "V": 1.15, "Y": 1.15,
        "9": 1.1, "3": 1.1, "2": 1.1,
        "1": 1.05, "7": 1.05,
    }

    def _compute_sample_weight(self, label: str, is_synthetic: bool) -> float:
        """Return a weight >1.0 if label contains hard characters.
        Synthetic samples get a small penalty (0.9) to avoid over‑emphasis.
        """
        label = label.upper()
        score = sum(self.HARD_CHAR_WEIGHTS.get(c, 0.0) for c in label)
        weight = 1.0 + 0.08 * score
        if is_synthetic:
            weight *= 0.9
        return min(weight, 3.0)

    # --------------------------------------------------------------------------
    # Existing methods (minimally modified)
    # --------------------------------------------------------------------------
    def _load_or_create_split(
        self,
        all_tracks: List[str],
        split_ratio: float
    ) -> Tuple[List[str], List[str]]:
        
        if self.full_train:
            print("📌 FULL TRAIN MODE: Using all tracks for training (no validation split).")
            return all_tracks, []

        # ==========================================================
        # ALWAYS USE EXISTING VALIDATION FILE IF IT EXISTS
        # ==========================================================
        if os.path.exists(self.val_split_file):
            print(f"📂 Loading split from '{self.val_split_file}'...")
            
            with open(self.val_split_file, 'r') as f:
                val_ids = set(json.load(f))
            
            val_tracks = []
            train_tracks = []
            
            # FIX: ONLY Scenario-B tracks can enter validation
            for t in all_tracks:
                track_name = os.path.basename(t)
                is_scenario_b = "Scenario-B" in t
                
                # ONLY Scenario-B tracks can be in validation
                if is_scenario_b and track_name in val_ids:
                    val_tracks.append(t)
                else:
                    train_tracks.append(t)
            
            print(f"✅ Fixed validation set loaded: {len(val_tracks)} tracks (from {self.val_split_file})")
            
            # Check if Scenario-B exists in validation (warning only, NO recreation)
            scenario_b_in_val = any("Scenario-B" in t for t in val_tracks)
            if not scenario_b_in_val:
                print(f"⚠️ Warning: Validation set has no Scenario-B tracks, but keeping original split anyway.")
            
            return train_tracks, val_tracks

        # ==========================================================
        # CREATE SPLIT ONLY FIRST TIME
        # ==========================================================
        print("⚠️ Validation split file not found. Creating new split...")
        
        scenario_b_tracks = [t for t in all_tracks if "Scenario-B" in t]
        
        if not scenario_b_tracks:
            print("⚠️ Warning: No 'Scenario-B' folder found. Using random split.")
            scenario_b_tracks = all_tracks
        
        val_size = max(1, int(len(scenario_b_tracks) * (1 - split_ratio)))
        
        random.Random(self.seed).shuffle(scenario_b_tracks)
        
        val_tracks = scenario_b_tracks[:val_size]
        
        val_set = set(val_tracks)
        
        train_tracks = [t for t in all_tracks if t not in val_set]
        
        os.makedirs(os.path.dirname(self.val_split_file), exist_ok=True)
        
        with open(self.val_split_file, 'w') as f:
            json.dump([os.path.basename(t) for t in val_tracks], f, indent=2)
        
        print(f"✅ Created validation split: {len(val_tracks)} tracks")
        
        return train_tracks, val_tracks

    def _index_samples(self, tracks: List[str]) -> None:
        """Index all samples from selected tracks and store corresponding weights."""
        for track_path in tqdm(tracks, desc=f"Indexing {self.mode}"):
            json_path = os.path.join(track_path, "annotations.json")
            if not os.path.exists(json_path):
                continue
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    data = data[0]
                label = data.get('plate_text', data.get('license_plate', data.get('text', '')))
                if not label:
                    continue
                
                track_id = os.path.basename(track_path)
                
                lr_files = sorted(
                    glob.glob(os.path.join(track_path, "lr-*.png")) +
                    glob.glob(os.path.join(track_path, "lr-*.jpg"))
                )
                hr_files = sorted(
                    glob.glob(os.path.join(track_path, "hr-*.png")) +
                    glob.glob(os.path.join(track_path, "hr-*.jpg"))
                )
                
                # --- Real LR sample ---
                self.samples.append({
                    'paths': lr_files,
                    'label': label,
                    'is_synthetic': False,
                    'track_id': track_id
                })
                weight_real = self._compute_sample_weight(label, is_synthetic=False)
                self.sample_weights.append(weight_real)
                
                # --- Synthetic LR sample (only in training mode) ---
                if self.mode == 'train':
                    if random.random() < 0.65:
                        self.samples.append({
                            'paths': hr_files,
                            'label': label,
                            'is_synthetic': True,
                            'track_id': track_id
                        })
                        weight_syn = self._compute_sample_weight(label, is_synthetic=True)
                        self.sample_weights.append(weight_syn)
            except Exception:
                pass

    def _index_test_samples(self, tracks: List[str]) -> None:
        """Index test samples without labels (weights are not used, but we store 1.0 for consistency)."""
        for track_path in tqdm(tracks, desc="Indexing test"):
            track_id = os.path.basename(track_path)
            lr_files = sorted(
                glob.glob(os.path.join(track_path, "lr-*.png")) +
                glob.glob(os.path.join(track_path, "lr-*.jpg"))
            )
            if lr_files:
                self.samples.append({
                    'paths': lr_files,
                    'label': '',
                    'is_synthetic': False,
                    'track_id': track_id
                })
                self.sample_weights.append(1.0)   # dummy, not used in test loader

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str, str]:
        item = self.samples[idx]
        img_paths = item['paths']
        label = item['label']
        is_synthetic = item['is_synthetic']
        track_id = item['track_id']
        
        images_list = []
        for p in img_paths:
            image = cv2.imread(p, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if is_synthetic and self.degrade:
                image = self.degrade(image=image)['image']
            image = self.transform(image=image)['image']
            images_list.append(image)

        images_tensor = torch.stack(images_list, dim=0)
        
        if self.is_test:
            target = [0]
            target_len = 1
        else:
            target = [self.char2idx[c] for c in label if c in self.char2idx]
            if len(target) == 0:
                target = [0]
            target_len = len(target)
            
        return images_tensor, torch.tensor(target, dtype=torch.long), target_len, label, track_id

    @staticmethod
    def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[str, ...], Tuple[str, ...]]:
        images, targets, target_lengths, labels_text, track_ids = zip(*batch)
        images = torch.stack(images, 0)
        targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long)
        return images, targets_padded, target_lengths, labels_text, track_ids