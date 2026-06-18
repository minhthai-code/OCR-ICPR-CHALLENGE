"""Post-processing utilities for OCR decoding.

Upgraded with:
- Log‑space beam search (stable probabilities)
- Weighted multi‑pattern syntax scoring (soft priors)
- Confusion priors with direction and length handling
- Beam‑expansion bias (early guidance)
- Fixed confidence lookup and beam pruning
"""

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import torch

# ============================================================
# 1. Constants and heuristics
# ============================================================
LETTER_TO_DIGIT = {
    'O': '0', 'Q': '0', 'D': '0',
    'I': '1', 'L': '1',
    'Z': '2',
    'S': '5',
    'G': '6',
    'B': '8',
}

DIGIT_TO_LETTER = {
    '0': 'O', '1': 'I', '2': 'Z',
    '5': 'S', '6': 'G', '8': 'B',
}

# Weighted plate patterns (prior probabilities, sum = 1)
WEIGHTED_PLATE_PATTERNS = [
    ("LLLNNNN", 0.6),   # old Brazilian format
    ("LLLNLNN", 0.4),   # Mercosul format
]
# Legacy list for old functions
PLATE_PATTERNS = [p for p, _ in WEIGHTED_PLATE_PATTERNS]

# Directional confusion priors (OCR ambiguity weights)
CONFUSION_PRIORS = {
    # letter → digit
    ('O', '0'): 0.92, ('D', '0'): 0.72, ('Q', '0'): 0.70,
    ('I', '1'): 0.88, ('L', '1'): 0.80,
    ('Z', '2'): 0.85, ('S', '5'): 0.76, ('G', '6'): 0.70, ('B', '8'): 0.80,
    # digit → letter
    ('0', 'O'): 0.90, ('0', 'D'): 0.68, ('0', 'Q'): 0.65,
    ('1', 'I'): 0.82, ('2', 'Z'): 0.80, ('5', 'S'): 0.72,
    ('6', 'G'): 0.68, ('8', 'B'): 0.78,
    # digit‑digit / letter‑letter
    ('1', '7'): 0.65, ('7', '1'): 0.60,
    ('5', '6'): 0.55, ('6', '5'): 0.53,
}

# Reranking weights (tunable)
SYNTAX_WEIGHT = 0.5
CONFUSION_WEIGHT = 0.3
EXPANSION_BIAS_WEIGHT = 0.1


# ============================================================
# 2. Helper functions (legacy hard correction)
# ============================================================
def _apply_pattern(text: str, pattern: str) -> Optional[str]:
    if len(text) != len(pattern):
        return None
    corrected = []
    for ch, p in zip(text, pattern):
        if p == 'L':
            if ch.isdigit():
                ch = DIGIT_TO_LETTER.get(ch, ch)
            if not ch.isalpha():
                return None
            corrected.append(ch.upper())
        elif p == 'N':
            if ch.isalpha():
                ch = LETTER_TO_DIGIT.get(ch.upper(), ch)
            if not ch.isdigit():
                return None
            corrected.append(ch)
        else:
            return None
    return "".join(corrected)


def apply_layout_correction(text: str) -> str:
    text = text.upper()
    candidates = [_apply_pattern(text, p) for p in PLATE_PATTERNS]
    candidates = [c for c in candidates if c is not None]
    if not candidates:
        return text
    return candidates[1] if len(candidates) == 2 else candidates[0]


# ============================================================
# 3. Syntax scoring (weighted, soft)
# ============================================================
def compute_syntax_score(
    text: str,
    patterns: Optional[Union[List[str], List[Tuple[str, float]]]] = None
) -> float:
    """
    Weighted syntax score. Returns max over patterns of (normalized match * weight).
    Range: [-1, 1].
    """
    if patterns is None:
        patterns = WEIGHTED_PLATE_PATTERNS
    text = text.upper()
    best = -1.0
    for item in patterns:
        if isinstance(item, tuple):
            pattern, weight = item
        else:
            pattern, weight = item, 1.0
        if len(text) != len(pattern):
            continue
        score = 0.0
        for ch, p in zip(text, pattern):
            if p == 'L':
                score += 1 if ch.isalpha() else -1
            else:  # 'N'
                score += 1 if ch.isdigit() else -1
        norm = (score / len(pattern)) * weight
        best = max(best, norm)
    return best if best > -1.0 else -1.0


# ============================================================
# 4. Confusion scoring (directional, length‑tolerant)
# ============================================================
def compute_confusion_score(text: str, pattern: Optional[str] = None) -> float:
    """
    Returns a soft bonus based on directional confusion priors.
    Handles length mismatch by padding with '<EOS>' (weight 0).
    """
    if pattern is None:
        pattern = "L" * len(text)  # fallback – kept for API
    max_len = max(len(text), len(pattern))
    total = 0.0
    for i in range(max_len):
        ch = text[i] if i < len(text) else '<EOS>'
        p = pattern[i] if i < len(pattern) else '<EOS>'
        if (ch, p) in CONFUSION_PRIORS:
            total += CONFUSION_PRIORS[(ch, p)]
    max_possible = max_len * max(CONFUSION_PRIORS.values()) if CONFUSION_PRIORS else 1.0
    return total / max_possible if max_possible > 0 else 0.0


# ============================================================
# 5. Legacy greedy decoding (unchanged)
# ============================================================
def _decode_slot_sequence(probs: torch.Tensor, idx2char: Dict[int, str]) -> List[Tuple[str, float]]:
    max_probs, indices = probs.max(dim=-1)
    results = []
    for b in range(indices.size(0)):
        chars, confs = [], []
        for s in range(indices.size(1)):
            idx = int(indices[b, s].item())
            if idx == 0:
                continue
            ch = idx2char.get(idx, '')
            if ch:
                chars.append(ch)
                confs.append(float(max_probs[b, s].item()))
        pred = "".join(chars)
        conf = float(np.mean(confs)) if confs else 0.0
        if conf < 0.85:
            pred = apply_layout_correction(pred)
        results.append((pred, conf))
    return results


def decode_with_confidence(
    logits: torch.Tensor,
    idx2char: Dict[int, str],
    use_layout: bool = False,
    decode_mode: str = "greedy",
    beam_width: int = 5,
) -> List[Tuple[str, float]]:
    if decode_mode == "reranked_beam":
        return reranked_beam_search_decode(logits, idx2char, beam_width=beam_width)
    elif decode_mode == "beam":
        probs = torch.softmax(logits, dim=-1)
        return beam_search_decode(probs, idx2char, beam_width=beam_width)
    # greedy
    probs = torch.softmax(logits, dim=-1)
    max_probs, indices = probs.max(dim=-1)
    results = []
    for b in range(indices.size(0)):
        chars, confs = [], []
        for s in range(indices.size(1)):
            idx = int(indices[b, s].item())
            if idx == 0:
                continue
            ch = idx2char.get(idx, '')
            if ch:
                chars.append(ch)
                confs.append(float(max_probs[b, s].item()))
        pred = "".join(chars)
        conf = float(np.mean(confs)) if confs else 0.0
        if use_layout and conf < 0.85:
            pred = apply_layout_correction(pred)
        results.append((pred, conf))
    return results


# ============================================================
# 6. Improved reranked beam search (all fixes applied)
# ============================================================
def reranked_beam_search_decode(
    logits: torch.Tensor,
    idx2char: Dict[int, str],
    beam_width: int = 5,
    syntax_weight: float = SYNTAX_WEIGHT,
    confusion_weight: float = CONFUSION_WEIGHT,
    expansion_bias_weight: float = EXPANSION_BIAS_WEIGHT,
    patterns: Optional[Union[List[str], List[Tuple[str, float]]]] = None,
) -> List[Tuple[str, float]]:
    if logits.dim() != 3:
        raise ValueError(f"Expected 3D logits, got {logits.shape}")

    if patterns is None:
        patterns = WEIGHTED_PLATE_PATTERNS
    # Normalize to weighted list
    if patterns and isinstance(patterns[0], str):
        patterns = [(p, 1.0) for p in patterns]
    pattern_strings = [p for p, _ in patterns]
    max_pattern_len = max(len(p) for p in pattern_strings)  # FIX 2

    log_probs = torch.log_softmax(logits, dim=-1)
    log_probs_np = log_probs.detach().cpu().numpy()
    B, S, C = log_probs_np.shape
    results = []

    for b in range(B):
        beams = [(tuple(), 0.0)]  # (seq, log_visual_score)

        for s in range(S):
            new_beams = {}
            slot_log_probs = log_probs_np[b, s]
            top_idx = np.argsort(slot_log_probs)[::-1][:beam_width]

            for seq, log_score in beams:
                for c in top_idx:
                    new_seq = seq if c == 0 else seq + (int(c),)
                    new_log_score = log_score + slot_log_probs[c]

                    # Expansion bias (syntax only, after 3 chars)
                    if len(new_seq) >= 3:
                        partial = "".join([idx2char.get(i, '') for i in new_seq if i != 0])
                        if len(partial) <= max_pattern_len:   # FIX 2 applied
                            syn = compute_syntax_score(partial, patterns)
                            new_log_score += expansion_bias_weight * syn

                    # Keep best score per sequence (max)
                    key = new_seq
                    if key in new_beams:
                        new_beams[key] = max(new_beams[key], new_log_score)
                    else:
                        new_beams[key] = new_log_score

            # Prune to beam_width
            beams = sorted(new_beams.items(), key=lambda x: x[1], reverse=True)[:beam_width]

        # Build a dict for fast visual score lookup
        visual_scores = {seq: score for seq, score in beams}

        # Final reranking
        # ============================================================
        best_seq = None
        best_total = -float('inf')
        best_visual = -float('inf')
        best_visual_seq = None

        ref_pattern = pattern_strings[0]
        ref_len = len(ref_pattern)

        for seq, vis_score in beams:
            chars = [idx2char.get(i, '') for i in seq if i != 0]
            text = "".join(chars)

            # ❗ Do not discard early candidates too aggressively
            if len(text) == 0:
                continue

            syn_score = compute_syntax_score(text, patterns)
            conf_score = compute_confusion_score(text, ref_pattern)
            total = vis_score + syntax_weight * syn_score + confusion_weight * conf_score

            if total > best_total:
                best_total = total
                best_seq = seq

            if vis_score > best_visual:
                best_visual = vis_score
                best_visual_seq = seq

        # ============================================================
        # HARD FALLBACK SAFETY (CRITICAL FIX)
        # ============================================================
        if best_seq is None:
            best_seq = best_visual_seq

        if best_seq is None and len(beams) > 0:
            best_seq = beams[0][0]

        if best_seq is None:
            # absolute last fallback (prevents crash forever)
            results.append(("", 0.0))
            continue

        text = "".join([idx2char.get(i, '') for i in best_seq if i != 0])

        # ============================================================
        # CONFIDENCE (visual only)
        # ============================================================
        vis = visual_scores.get(best_seq, best_visual)
        num_tokens = len([i for i in best_seq if i != 0])
        if num_tokens > 0:
            avg_log_prob = vis / num_tokens
            confidence = float(np.exp(avg_log_prob))
            confidence = min(max(confidence, 0.0), 1.0)
        else:
            confidence = 0.0

        results.append((text, confidence))

    return results


# ============================================================
# 7. Legacy beam search (unchanged)
# ============================================================
def beam_search_decode(preds: torch.Tensor, idx2char: Dict[int, str], beam_width: int = 5) -> List[Tuple[str, float]]:
    if preds.dim() != 3:
        raise ValueError(f"Expected 3D tensor, got {preds.shape}")
    probs = preds.exp().detach().cpu().numpy() if preds.min().item() < 0 else preds.detach().cpu().numpy()
    B, S, C = probs.shape
    results = []
    for b in range(B):
        beams = [(tuple(), 1.0)]
        for s in range(S):
            new_beams = {}
            slot_probs = probs[b, s]
            top_idx = np.argsort(slot_probs)[::-1][:beam_width]
            for seq, score in beams:
                for c in top_idx:
                    new_seq = seq if c == 0 else seq + (int(c),)
                    new_score = score * float(slot_probs[c])
                    if new_seq in new_beams:
                        new_beams[new_seq] += new_score
                    else:
                        new_beams[new_seq] = new_score
            beams = sorted(new_beams.items(), key=lambda x: x[1], reverse=True)[:beam_width]
        best_seq, best_score = beams[0]
        pred_chars = [idx2char.get(i, '') for i in best_seq if i != 0]
        pred_text = "".join(pred_chars)
        confidence = float(best_score ** (1.0 / max(len(best_seq), 1))) if best_seq else 0.0
        if confidence < 0.85:
            pred_text = apply_layout_correction(pred_text)
        results.append((pred_text, confidence))
    return results