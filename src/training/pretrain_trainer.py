"""Pretraining trainer for Stage 1 with MVCP fusion."""
import copy
import os
import random
from typing import Dict, List, Tuple, Any
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from src.utils.postprocess import decode_with_confidence


class PretrainTrainer:
    """Trainer for stage-1 pretraining with MVCP support."""

    def __init__(self, model, train_loader, val_loader, config, idx2char):
        print("🔥 INITIALIZING PRETRAIN TRAINER (DEEPCOPY VERSION)")
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.idx2char = idx2char
        self.device = config.DEVICE

        # =========================================================
        # EMA teacher is a DEEPCOPY of the student.
        # =========================================================
        print("🔥 Creating teacher via deepcopy...")
        self.teacher_model = copy.deepcopy(model).to(self.device)
        for p in self.teacher_model.parameters():
            p.requires_grad = False
        self.teacher_model.eval()
        print("🔥 Teacher deepcopy complete.")

        # Loss weights
        self.ema_decay = getattr(config, "EMA_DECAY", 0.999)
        self.distill_weight = getattr(config, "DISTILL_WEIGHT", 0.05)
        self.distill_temperature = getattr(config, "DISTILL_TEMPERATURE", 2.0)
        self.consistency_weight = getattr(config, "CONSISTENCY_LOSS_WEIGHT", 0.02)

        self.risk_loss_weight = getattr(config, "RISK_LOSS_WEIGHT", 0.02)
        self.refine_loss_weight = getattr(config, "REFINE_LOSS_WEIGHT", 0.5)

        print(
            f"📌 EMA Decay: {self.ema_decay} | "
            f"Distill Weight: {self.distill_weight} | "
            f"Consistency Weight: {self.consistency_weight} | "
            f"Risk Weight: {self.risk_loss_weight} | "
            f"Refine Weight: {self.refine_loss_weight} | "
            f"Temp: {self.distill_temperature}"
        )

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.LEARNING_RATE,
            pct_start=0.2,
            div_factor=8,
            final_div_factor=20,
            steps_per_epoch=len(train_loader),
            epochs=config.EPOCHS,
        )

        self.scaler = GradScaler()

        self.best_acc = 0.0
        self.best_val_loss = float("inf")
        self.current_epoch = 0
        self.epochs_no_improve = 0
        self.patience = 10  # Match old run
        self.ramp_epochs = 10
        self.ema_start_epoch = 5

        self.char_confusions = Counter()
        self.position_confusions = Counter()

        # Hard confusion groups for CASLS soft targets
        self.confusion_groups = [
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
        self.casls_smoothing = 0.04
        self.casls_confusion_boost = 0.12

        self.log_refinement_stats = getattr(config, "LOG_REFINEMENT_STATS", False)

    # ------------------------------------------------------------------
    #  Checkpoint methods (resume support)
    # ------------------------------------------------------------------
    def _get_output_path(self, filename: str) -> str:
        output_dir = getattr(self.config, "OUTPUT_DIR", "results")
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, filename)

    def _get_exp_name(self) -> str:
        return getattr(self.config, "EXPERIMENT_NAME", "baseline")

    def save_checkpoint(self, path: str, epoch: int, is_best: bool = False) -> None:
        """Save full training state for resumption."""
        ckpt = {
            "epoch": epoch,  # next epoch to run
            "student": self.model.state_dict(),
            "teacher": self.teacher_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best_acc": self.best_acc,
            "best_val_loss": self.best_val_loss,
            "epochs_no_improve": self.epochs_no_improve,
            "config": {
                "EPOCHS": self.config.EPOCHS,
                "LEARNING_RATE": self.config.LEARNING_RATE,
                "BATCH_SIZE": self.config.BATCH_SIZE,
                "WEIGHT_DECAY": self.config.WEIGHT_DECAY,
            },
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        torch.save(ckpt, path)
        print(f"✅ Saved checkpoint: {path}")

        if is_best:
            best_path = self._get_output_path(f"{self._get_exp_name()}_best.pth")
            torch.save(ckpt, best_path)
            print(f"⭐ Saved best checkpoint: {best_path}")

    def load_checkpoint(self, path: str, strict: bool = True) -> int:
        """Load full training state. Returns next epoch index (0‑based)."""
        print(f"🔄 Loading checkpoint: {path}")
        # Trusted checkpoint – disable weights_only to allow NumPy state etc.
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(ckpt["student"], strict=strict)
        self.teacher_model.load_state_dict(ckpt["teacher"], strict=strict)

        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler"])

        self.best_acc = ckpt.get("best_acc", 0.0)
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        self.epochs_no_improve = ckpt.get("epochs_no_improve", 0)

        # ---- Safe RNG restore ----
        rng_state = ckpt.get("rng_state")
        if rng_state is not None:
            if "python" in rng_state:
                random.setstate(rng_state["python"])
            if "numpy" in rng_state:
                np.random.set_state(rng_state["numpy"])
            if "torch" in rng_state and rng_state["torch"] is not None:
                torch_state = rng_state["torch"]
                try:
                    if isinstance(torch_state, torch.ByteTensor):
                        pass
                    elif isinstance(torch_state, torch.Tensor):
                        torch_state = torch_state.to(dtype=torch.uint8)
                    else:
                        torch_state = torch.from_numpy(np.array(torch_state, dtype=np.uint8))
                    torch_state = torch_state.contiguous()
                    if torch_state.numel() == torch.get_rng_state().numel():
                        torch.set_rng_state(torch_state)
                except Exception as e:
                    print(f"⚠️ Skipping torch RNG restore: {e}")
            if torch.cuda.is_available() and "cuda" in rng_state and rng_state["cuda"] is not None:
                try:
                    cuda_states = rng_state["cuda"]
                    if isinstance(cuda_states, list) and len(cuda_states) == torch.cuda.device_count():
                        torch.cuda.set_rng_state_all(cuda_states)
                except Exception as e:
                    print(f"⚠️ Could not restore CUDA RNG state: {e}")
        # ---- End safe restore ----

        start_epoch = int(ckpt.get("epoch", 0))
        print(f"✅ Resume ready. Next epoch: {start_epoch + 1}")
        return start_epoch

    # ------------------------------------------------------------------
    #  Utility methods
    # ------------------------------------------------------------------
    def _get_ramp_factor(self):
        return min(1.0, self.current_epoch / self.ramp_epochs)

    def _decode_predictions(self, logits: torch.Tensor):
        return decode_with_confidence(
            logits,
            self.idx2char,
            use_layout=self.config.LAYOUT_AWARE_DECODING,
            decode_mode=self.config.DECODE_MODE,
            beam_width=self.config.BEAM_WIDTH,
        )

    def _majority_vote_predictions(self, frame_logits: torch.Tensor) -> List[str]:
        B, F, T, C = frame_logits.shape
        preds = []
        for b in range(B):
            frame_preds = frame_logits[b].argmax(dim=-1)
            voted_chars = []
            for t in range(T):
                votes = frame_preds[:, t].tolist()
                most_common = max(set(votes), key=votes.count)
                if most_common != 0:
                    voted_chars.append(self.idx2char[most_common])
            preds.append(''.join(voted_chars))
        return preds

    def _update_confusion_stats(self, gt_text: str, pred_text: str):
        gt_text = str(gt_text).upper()
        pred_text = str(pred_text).upper()
        for pos, (g, p) in enumerate(zip(gt_text, pred_text)):
            if g != p:
                self.char_confusions[(g, p)] += 1
                self.position_confusions[(pos, g, p)] += 1
        if len(gt_text) != len(pred_text):
            self.position_confusions[("LEN", len(gt_text), len(pred_text))] += 1

    @torch.no_grad()
    def _update_teacher(self):
        if self.current_epoch < self.ema_start_epoch:
            return
        for t_param, s_param in zip(self.teacher_model.parameters(), self.model.parameters()):
            t_param.data.mul_(self.ema_decay).add_(s_param.data, alpha=1.0 - self.ema_decay)
        for t_buf, s_buf in zip(self.teacher_model.buffers(), self.model.buffers()):
            t_buf.copy_(s_buf)

    def _build_confusion_soft_targets(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        smoothing: float = 0.04,
        confusion_boost: float = 0.12,
    ) -> torch.Tensor:
        N, C = logits.shape
        device = logits.device

        soft_targets = torch.full((N, C), smoothing / C, device=device, dtype=torch.float32)
        soft_targets.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)

        for group in self.confusion_groups:
            ids = [self.config.CHAR2IDX[ch] for ch in group if ch in self.config.CHAR2IDX]
            if len(ids) < 2:
                continue
            for cid in ids:
                mask = (targets == cid)
                if not mask.any():
                    continue
                n_others = len(ids) - 1
                boost_per_other = confusion_boost / n_others
                for other in ids:
                    if other != cid:
                        soft_targets[mask, other] += boost_per_other
                soft_targets[mask, cid] -= confusion_boost

        soft_targets = torch.clamp(soft_targets, min=0.0)
        soft_targets = soft_targets / soft_targets.sum(dim=-1, keepdim=True)
        return soft_targets

    # ------------------------------------------------------------------
    #  Training and validation loops (fully implemented)
    # ------------------------------------------------------------------
    def train_one_epoch(self) -> float:
        self.model.train()
        epoch_loss = 0.0
        epoch_risk_loss = 0.0
        epoch_refine_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Ep {self.current_epoch + 1}/{self.config.EPOCHS}")

        ramp_factor = self._get_ramp_factor()

        for images, targets, _, _, _ in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast("cuda"):
                use_refinement = getattr(self.model, "use_refinement", False)

                if use_refinement:
                    outputs = self.model(
                        images,
                        return_frame_logits=True,
                        return_aux=True,
                    )
                    base_logits = outputs["base_logits"]
                    refined_logits = outputs["refined_logits"]
                    student_logits = refined_logits if refined_logits is not None else base_logits
                    frame_logits = outputs["frame_logits"]
                    risk_scores = outputs["risk_scores"]
                else:
                    student_logits, frame_logits = self.model(images, return_frame_logits=True)
                    base_logits = None
                    refined_logits = None
                    risk_scores = None

                # Teacher forward in the same autocast context (FP16)
                with torch.no_grad():
                    teacher_logits, _ = self.teacher_model(images, return_frame_logits=True)

                if targets.size(1) < student_logits.size(1):
                    targets = F.pad(targets, (0, student_logits.size(1) - targets.size(1)), value=0)
                else:
                    targets = targets[:, :student_logits.size(1)]

                flat_logits = student_logits.reshape(-1, student_logits.size(-1))
                flat_targets = targets.reshape(-1)

                valid_mask = (flat_targets != 0).float()

                soft_targets = self._build_confusion_soft_targets(
                    flat_logits,
                    flat_targets,
                    smoothing=self.casls_smoothing,
                    confusion_boost=self.casls_confusion_boost,
                )

                log_probs = F.log_softmax(flat_logits, dim=-1)
                per_token_loss = -(soft_targets * log_probs).sum(dim=-1)
                slot_loss = (per_token_loss * valid_mask).sum() / (valid_mask.sum() + 1e-6)

                risk_loss = torch.tensor(0.0, device=self.device)
                if risk_scores is not None and base_logits is not None:
                    with torch.no_grad():
                        base_preds = base_logits.argmax(dim=-1)
                        token_error = (base_preds != targets) & (targets != 0)
                        token_error = token_error.float()
                    risk_loss = F.binary_cross_entropy_with_logits(risk_scores.squeeze(-1), token_error)

                refine_loss = torch.tensor(0.0, device=self.device)
                if refined_logits is not None and self.refine_loss_weight > 0:
                    flat_refined = refined_logits.reshape(-1, refined_logits.size(-1))
                    refine_loss = F.cross_entropy(flat_refined, flat_targets, ignore_index=0)

                # 🔧 FIX: Only compute consistency loss if frame_logits exists (multi‑frame models)
                if frame_logits is not None:
                    frame_log_probs = F.log_softmax(frame_logits, dim=-1)
                    fused_probs = F.softmax(student_logits.unsqueeze(1), dim=-1)
                    consistency_loss = F.kl_div(
                        frame_log_probs, fused_probs, reduction="batchmean"
                    ) * (self.consistency_weight * ramp_factor)
                else:
                    consistency_loss = torch.tensor(0.0, device=self.device, dtype=student_logits.dtype)

                if self.current_epoch >= self.ema_start_epoch:
                    T = self.distill_temperature
                    student_log_probs = F.log_softmax(student_logits / T, dim=-1)
                    teacher_probs = F.softmax(teacher_logits / T, dim=-1)
                    distill_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (T ** 2) * (self.distill_weight * ramp_factor)
                else:
                    distill_loss = torch.tensor(0.0, device=self.device)

                loss = (
                    slot_loss + consistency_loss + distill_loss +
                    self.risk_loss_weight * risk_loss +
                    self.refine_loss_weight * refine_loss
                )

            if not torch.isfinite(loss):
                print("⚠️ NaN/Inf loss detected. Skipping batch.")
                self.optimizer.zero_grad()
                continue

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            self._update_teacher()

            epoch_loss += loss.item()
            epoch_risk_loss += risk_loss.item()
            epoch_refine_loss += refine_loss.item()

            refine_gap = 0.0
            avg_risk = 0.0
            if refined_logits is not None and base_logits is not None:
                refine_gap = (refined_logits - base_logits).abs().mean().item()
            if risk_scores is not None:
                avg_risk = torch.sigmoid(risk_scores).mean().item()

            pbar.set_postfix(
                {
                    "loss": loss.item(),
                    "slot": slot_loss.item(),
                    "risk": risk_loss.item(),
                    "refine": refine_loss.item(),
                    "consist": consistency_loss.item(),
                    "distill": distill_loss.item(),
                    "gap": refine_gap,
                    "risk_mean": avg_risk,
                    "ramp": f"{ramp_factor:.2f}",
                    "lr": self.scheduler.get_last_lr()[0],
                }
            )

        avg_loss = epoch_loss / len(self.train_loader)
        avg_risk = epoch_risk_loss / len(self.train_loader)
        avg_refine = epoch_refine_loss / len(self.train_loader)
        print(f"  📊 Epoch avg: Loss={avg_loss:.4f} | Risk={avg_risk:.4f} | Refine={avg_refine:.4f}")
        return avg_loss

    def validate(self) -> Tuple[Dict[str, float], List[str], List[Dict[str, Any]]]:
        if self.val_loader is None:
            return {"loss": 0.0, "acc": 0.0, "mvcp_acc": 0.0}, [], []

        self.char_confusions.clear()
        self.position_confusions.clear()

        val_loss, total_correct, total_samples = 0.0, 0, 0
        total_correct_mvcp = 0

        submission_data: List[str] = []
        detailed_results: List[Dict[str, Any]] = []

        model_to_validate = self.model
        model_tag = "STUDENT"
        model_to_validate.eval()

        display_epoch = self.current_epoch + 1
        should_log = (display_epoch % 2 != 0) or (display_epoch % 10 == 0)
        MAX_LOG = 20 if (display_epoch % 10 == 0) else 10

        use_refined_validation = getattr(self.model, "use_refinement", False)
        if use_refined_validation:
            print(f"🔍 VALIDATION USING REFINED LOGITS (epoch {display_epoch})")
        else:
            print(f"🔍 VALIDATION USING BASE LOGITS (epoch {display_epoch})")

        with torch.no_grad():
            for images, targets, _, labels_text, track_ids in self.val_loader:
                images, targets = images.to(self.device), targets.to(self.device)

                if use_refined_validation:
                    fused_logits, frame_logits = model_to_validate(
                        images,
                        return_frame_logits=True,
                        use_refinement=True
                    )
                else:
                    fused_logits, frame_logits = model_to_validate(images, return_frame_logits=True)

                if targets.size(1) < fused_logits.size(1):
                    targets_pad = F.pad(targets, (0, fused_logits.size(1) - targets.size(1)), value=0)
                else:
                    targets_pad = targets[:, :fused_logits.size(1)]

                loss = F.cross_entropy(
                    fused_logits.reshape(-1, fused_logits.size(-1)),
                    targets_pad.reshape(-1),
                    ignore_index=0,
                )
                val_loss += loss.item()

                decoded_list = self._decode_predictions(fused_logits)
                # 🔧 FIX: Only compute MVCP predictions if frame_logits exists
                if frame_logits is not None:
                    mvcp_preds = self._majority_vote_predictions(frame_logits)
                else:
                    mvcp_preds = [""] * len(labels_text)

                for i, (pred_text, conf) in enumerate(decoded_list):
                    gt_text = labels_text[i]
                    track_id = track_ids[i]
                    final_pred = pred_text[:7].upper()
                    mvcp_pred = mvcp_preds[i][:7].upper() if i < len(mvcp_preds) else ""

                    detailed_results.append({
                        "track_id": track_id,
                        "gt_text": gt_text,
                        "final_pred": final_pred,
                        "mvcp_pred": mvcp_pred,
                        "conf": conf,
                    })

                    self._update_confusion_stats(gt_text, final_pred)

                    if should_log and total_samples < MAX_LOG:
                        refine_tag = " (refined)" if use_refined_validation else ""
                        print(
                            f"    🔍 [{model_tag}{refine_tag} Ep {display_epoch}] Track: {track_id} | "
                            f"GT: {gt_text} | Fused: {final_pred} (conf={conf:.2f}) | MVCP: {mvcp_pred}"
                        )

                    if final_pred == gt_text:
                        total_correct += 1
                    # 🔧 FIX: Only count MVCP correct if frame_logits exists
                    if frame_logits is not None and mvcp_pred == gt_text:
                        total_correct_mvcp += 1

                    submission_data.append(f"{track_id},{final_pred};{conf:.4f}")
                    total_samples += 1

        avg_val_loss = val_loss / len(self.val_loader)
        val_acc = (total_correct / total_samples) * 100 if total_samples > 0 else 0.0
        # 🔧 FIX: Compute mvcp_acc only if any MVCP predictions were made
        if total_correct_mvcp > 0:
            mvcp_acc = (total_correct_mvcp / total_samples) * 100 if total_samples > 0 else 0.0
        else:
            mvcp_acc = 0.0

        print(f"\n📊 Validation Results (Epoch {display_epoch}):")
        print(f"   Baseline (fused) accuracy : {val_acc:.2f}%")
        if frame_logits is not None:
            print(f"   MVCP (per‑frame vote)     : {mvcp_acc:.2f}%")
            if mvcp_acc > val_acc:
                print(f"   ✅ MVCP beats baseline by {mvcp_acc - val_acc:.2f}%")
        else:
            print("   MVCP (per‑frame vote)     : N/A (single‑frame model)")

        print("\n📊 Top Character Confusions (epoch):")
        for (gt, pred), count in self.char_confusions.most_common(20):
            print(f"  {gt} -> {pred}: {count}")

        print("\n📊 Top Position-Aware Confusions (epoch):")
        for (pos, gt, pred), count in self.position_confusions.most_common(20):
            if pos == "LEN":
                print(f"  LEN mismatch {gt} vs {pred}: {count}")
            else:
                print(f"  pos {pos}: {gt} -> {pred}: {count}")
        print("")

        metrics = {"loss": avg_val_loss, "acc": val_acc, "mvcp_acc": mvcp_acc}
        return metrics, submission_data, detailed_results

    def _log_detailed_results(self, detailed_results: List[Dict[str, Any]], new_acc: float) -> None:
        wrong = [d for d in detailed_results if d["final_pred"] != d["gt_text"]]
        correct = [d for d in detailed_results if d["final_pred"] == d["gt_text"]]

        print("\n" + "=" * 80)
        print(f"🎉 DETAILED RESULTS (Accuracy improved to {new_acc:.2f}%)")
        print(f"   Total validation samples: {len(detailed_results)}")
        print(f"   Wrong: {len(wrong)} | Correct: {len(correct)}")
        print("=" * 80)

        print(f"\n❌ WRONG PREDICTIONS (all {len(wrong)} cases):")
        for idx, item in enumerate(wrong, 1):
            print(f"  {idx:3d}. Track {item['track_id']:10s} | GT: {item['gt_text']} | Pred: {item['final_pred']} (conf={item['conf']:.4f}) | MVCP: {item['mvcp_pred']}")

        print(f"\n✅ CORRECT PREDICTIONS (first 10 of {len(correct)}):")
        for idx, item in enumerate(correct[:10], 1):
            print(f"  {idx:3d}. Track {item['track_id']:10s} | GT: {item['gt_text']} | Pred: {item['final_pred']} (conf={item['conf']:.4f}) | MVCP: {item['mvcp_pred']}")
        print("=" * 80 + "\n")

    def save_submission(self, submission_data: List[str]) -> None:
        exp_name = self._get_exp_name()
        filename = self._get_output_path(f"submission_{exp_name}.txt")
        with open(filename, "w") as f:
            f.write("\n".join(submission_data))
        print(f"📝 Saved {len(submission_data)} lines to {filename}")

    def save_model(self, path: str = None) -> None:
        """Convenience wrapper – saves only weights (not full state)."""
        if path is None:
            path = self._get_output_path(f"{self._get_exp_name()}_best.pth")
        torch.save(
            {
                "student": self.model.state_dict(),
                "teacher": self.teacher_model.state_dict(),
            },
            path,
        )

    # ------------------------------------------------------------------
    #  Modified fit() with resume support
    # ------------------------------------------------------------------
    def fit(self, start_epoch: int = 0) -> None:
        print(f"🚀 PRETRAIN START | Device: {self.device} | Epochs: {self.config.EPOCHS}")
        if getattr(self.model, "use_refinement", False):
            print("✨ Selective Correction ENABLED (Risk Gate + Confusion Refiner)")
            print(f"   Risk Loss Weight: {self.risk_loss_weight}")
            print(f"   Refine Loss Weight: {self.refine_loss_weight}")
        else:
            print("⚙️  Selective Correction DISABLED (baseline training)")

        for epoch in range(start_epoch, self.config.EPOCHS):
            self.current_epoch = epoch

            avg_train_loss = self.train_one_epoch()
            val_metrics, submission_data, detailed_results = self.validate()
            val_loss = val_metrics["loss"]
            val_acc = val_metrics["acc"]
            mvcp_acc = val_metrics.get("mvcp_acc", 0.0)
            current_lr = self.scheduler.get_last_lr()[0]

            print(
                f"Epoch {epoch + 1}/{self.config.EPOCHS}: "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Baseline Acc: {val_acc:.2f}% | "
                f"MVCP Acc: {mvcp_acc:.2f}% | "
                f"LR: {current_lr:.2e}"
            )

            improved = val_acc > self.best_acc and val_acc > 0

            if improved:
                print(f"  ✨ Accuracy Improved: {self.best_acc:.2f}% -> {val_acc:.2f}%")
                if val_acc > 80.0:
                    self._log_detailed_results(detailed_results, val_acc)

                self.best_acc = val_acc
                self.epochs_no_improve = 0

                if submission_data:
                    self.save_submission(submission_data)

                # Save as BEST (full checkpoint)
                latest_path = self._get_output_path(f"{self._get_exp_name()}_latest.pth")
                self.save_checkpoint(latest_path, epoch=epoch + 1, is_best=True)

            elif val_acc == 0 and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                latest_path = self._get_output_path(f"{self._get_exp_name()}_latest.pth")
                self.save_checkpoint(latest_path, epoch=epoch + 1, is_best=False)

            else:
                self.epochs_no_improve += 1
                print(f"  Patience: {self.epochs_no_improve}/{self.patience}")
                latest_path = self._get_output_path(f"{self._get_exp_name()}_latest.pth")
                self.save_checkpoint(latest_path, epoch=epoch + 1, is_best=False)

            if self.epochs_no_improve >= self.patience:
                print(f"🛑 Early stopping triggered at epoch {epoch + 1}")
                break

    # ------------------------------------------------------------------
    #  Inference methods
    # ------------------------------------------------------------------
    def predict(self, loader: DataLoader, use_mvcp: bool = True) -> List[Tuple[str, str, float]]:
        self.model.eval()
        results: List[Tuple[str, str, float]] = []
        use_refined = getattr(self.model, "use_refinement", False)

        with torch.no_grad():
            for images, _, _, _, track_ids in loader:
                images = images.to(self.device)
                if use_refined:
                    fused_logits, frame_logits = self.model(
                        images, return_frame_logits=True, use_refinement=True
                    )
                else:
                    fused_logits, frame_logits = self.model(images, return_frame_logits=True)

                if use_mvcp and frame_logits is not None:
                    mvcp_preds = self._majority_vote_predictions(frame_logits)
                    for i, track_id in enumerate(track_ids):
                        final_text = mvcp_preds[i][:7].upper() if i < len(mvcp_preds) else ""
                        results.append((track_id, final_text, 1.0))
                else:
                    decoded_list = self._decode_predictions(fused_logits)
                    for i, (pred_text, conf) in enumerate(decoded_list):
                        final_text = pred_text[:7].upper()
                        results.append((track_ids[i], final_text, conf))
        return results

    def predict_test(self, test_loader: DataLoader, output_filename: str = "submission_final.txt",
                     use_mvcp: bool = True) -> None:
        print(f"🔮 Running inference on test data (MVCP={use_mvcp})...")
        results = self.predict(test_loader, use_mvcp=use_mvcp)
        submission_data = [f"{track_id},{p_text};{c:.4f}" for track_id, p_text, c in results]
        output_path = self._get_output_path(output_filename)
        with open(output_path, "w") as f:
            f.write("\n".join(submission_data))
        print(f"✅ Saved {len(submission_data)} predictions to {output_path}")