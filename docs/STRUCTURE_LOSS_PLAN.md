# Plan: Structure-Aware Loss Function for Lyrics Generation

## Goal
Replace hardcoded limits (max words per line, max length) with a loss function that teaches the model proper lyric structure during training.

## Current Problem
- `generate.py:145-148` forces newlines every 10 words, ignoring model predictions
- The model already learns `<NEWLINE>` from training data but this is overridden

## Solution: Multi-Component Loss Function

```
L_total = L_ce + λ_line * L_line + λ_length * L_length
```

### Loss Components

| Component | Purpose | How it works |
|-----------|---------|--------------|
| **L_ce** | Primary cross-entropy | Standard loss with boosted weight (2x) for `<NEWLINE>` token |
| **L_line** | Line length guidance | Penalizes low `<NEWLINE>` probability after 7+ words; penalizes high probability before 3 words |
| **L_length** | Song length guidance | Sigmoid pressure that encourages `<EOS>` as total words approach target (~150) |

---

## Files to Modify

### 1. CREATE: `src/losses.py` (new file)
```python
class StructureAwareLoss(nn.Module):
    - __init__: Store vocab indices, hyperparameters
    - forward: Compute L_ce + λ_line * L_line + λ_length * L_length
    - _compute_line_length_loss: Track position-in-line, penalize missing newlines
    - _compute_length_guidance_loss: Sigmoid pressure for EOS
    - _compute_line_positions: Helper to count words since last <NEWLINE>
```

### 2. MODIFY: `src/train.py`
- Import `StructureAwareLoss` from losses
- Replace `nn.CrossEntropyLoss(ignore_index=0)` with new loss
- Change loss call to pass full tensors (not flattened)
- Pass vocabulary to `train_model` function
- Add TensorBoard logging for individual loss components

### 3. MODIFY: `src/generate.py`
- DELETE lines 145-148 (hardcoded line break logic)
- Remove `max_words_per_line` parameter
- Keep `max_length` as safety limit only (increase to 300)

### 4. MODIFY: `src/config.py`
Add structure loss hyperparameters

---

## Hyperparameters (Data-Driven from Training Set)

**Training data statistics (599 songs, 23,860 lines):**
- Line length: median=6, mean=6.6, std=3.7, 75% in range 5-8 words
- Song length: median=235, mean=261, std=146, 75% in range 172-305 words

| Parameter | Value | Description |
|-----------|-------|-------------|
| `lambda_line` | 0.1 | Weight for line length loss |
| `lambda_length` | 0.05 | Weight for total length loss |
| `newline_weight` | 2.0 | Boost CE weight for `<NEWLINE>` |
| `target_line_length` | **6.0** | Median words per line (from data) |
| `line_length_sigma` | **3.5** | Std dev from data |
| `target_song_length` | **235.0** | Median total words (from data) |
| `length_scale` | **75.0** | ~half the std dev for smooth pressure |

---

## Key Design Decisions

- **Line length penalty**: Uses soft Gaussian centered at 6 words - penalizes both too-short and too-long lines
- **Length guidance**: Sigmoid pressure (not hard cutoff) - model learns naturally when to end
- **Newline weighting**: 2x in cross-entropy ensures model pays attention to structure tokens
- **Conservative lambdas**: Start low (0.1, 0.05) to not dominate primary CE loss
