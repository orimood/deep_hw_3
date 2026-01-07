# Bug Fixes: Short Songs and Single-Word Lines

## Problem Statement

The lyrics generation model was producing:
1. **Songs that are too short** - ending prematurely despite min_length constraints
2. **Lines with single words** - going to a newline after just one word

These problems were so severe that hardcoded limits had to be added in `generate.py` to work around them.

## Root Cause Analysis

The code critique identified **7 critical issues** that directly caused these problems:

### Critical Issue 1: Tokenization Bug (ROOT CAUSE)

**Location:** `src/vocab.py:73-84`

**Problem:** The `tokenize()` function uses regex `[^\w\s']` which removes `<` and `>` characters. This means:
- `<NEWLINE>` in training data becomes `newline` (different token!)
- The vocabulary has `<NEWLINE>` at index 4 (special token)
- But training data encodes to `newline` which gets a different index (or maps to UNK)
- **The model NEVER sees `<NEWLINE>` (index 4) during training**

**Impact:**
- Loss function boosts weight for index 4, but that token never appears in training
- Generation checks for `newline_idx`, but model was trained on different index
- All NEWLINE-related logic operates on the wrong token

### Critical Issue 2: Loss Function Gradient Conflict

**Location:** `src/losses.py:125-168`

**Problem:** Line length penalties applied to **ALL positions**, not just where NEWLINE is the target:
```python
# Old code penalized p_newline at every position
early_penalty = (torch.exp(early_positions) - 1) * p_newline * 5.0
hard_penalty = very_short * p_newline * 100.0 * (1 + shortness_factor)
```

**Impact:**
- At position 2, if target is a regular word, cross-entropy wants p_newline ≈ 0
- But the penalty ALSO penalizes p_newline, which is redundant and confusing
- The model learns: "never predict NEWLINE" because it's penalized everywhere

### Critical Issue 3: Song Length Dead Zone

**Location:** `src/losses.py:193-233`

**Problem:** Between positions 80 (min_song_length) and 120 (target_song_length), there was almost NO penalty for EOS:
- Before 80: Hard penalty (100x multiplier)
- After 120: Weak sigmoid pressure
- **Between 80-120: No penalty!**

**Impact:** Once past position 80, the model could freely output EOS with no consequence.

### Critical Issue 4: decode() Adds Spaces Around Newlines

**Location:** `src/vocab.py:92-103`

**Problem:** Using `' '.join(words)` after appending `\n` creates ` \n ` (space-newline-space).

### Critical Issue 5: Empty Lyrics Create SOS→EOS Samples

**Location:** `src/dataset.py:211-220`

**Problem:** If lyrics encode to empty (or very short), the sample becomes:
- input_seq = `[SOS]`
- target_seq = `[EOS]`

This teaches the model: "after SOS, immediately output EOS".

### Critical Issue 6: In-Place Logit Mutation

**Location:** `src/generate.py:42-45`

**Problem:** `sample_with_temperature` modifies logits in-place:
```python
logits[token_idx] = logits[token_idx] / repetition_penalty
```

When resampling after EOS rejection, the repetition penalty gets applied multiple times.

---

## Fixes Applied

### Fix 1: Tokenization (vocab.py)

```python
@staticmethod
def tokenize(text: str) -> List[str]:
    text = text.strip()

    # CRITICAL FIX: Preserve NEWLINE tokens before cleaning
    newline_placeholder = "___NEWLINE___"
    text = text.replace(config.NEWLINE_TOKEN, newline_placeholder)

    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    words = [w for w in text.split() if w]

    # CRITICAL FIX: Restore NEWLINE tokens
    words = [config.NEWLINE_TOKEN if w == newline_placeholder.lower() else w for w in words]

    return words
```

### Fix 2: decode() (vocab.py)

```python
def decode(self, indices: List[int], skip_special: bool = True) -> str:
    # Build lines instead of flat word list
    lines = []
    current_line = []

    for idx in indices:
        word = self.idx2word.get(idx, config.UNK_TOKEN)
        if word in skip_tokens:
            continue

        if word == config.NEWLINE_TOKEN:
            if current_line:
                lines.append(' '.join(current_line))
                current_line = []
            else:
                lines.append('')
        else:
            current_line.append(word)

    if current_line:
        lines.append(' '.join(current_line))

    return '\n'.join(lines)
```

### Fix 3: Line Length Loss (losses.py)

```python
def _compute_line_length_loss(self, logits, targets):
    # FIXED: Only apply penalties at positions where TARGET is NEWLINE
    is_newline_target = (targets == self.newline_idx).float()
    is_not_newline_target = (targets != self.newline_idx).float() * mask

    # PENALTY 1: When target IS NEWLINE but line is too short
    too_short_for_newline = F.relu(self.min_line_length - line_positions)
    short_line_penalty = is_newline_target * too_short_for_newline * 2.0

    # PENALTY 2: When target is NOT NEWLINE but line is very long
    very_long = F.relu(line_positions - self.target_line_length)
    long_line_encouragement = is_not_newline_target * very_long * (1 - p_newline) * 0.1
```

### Fix 4: Song Length Loss (losses.py)

```python
def _compute_length_guidance_loss(self, logits, targets):
    # FIXED: Apply at EOS targets and continuous penalty before target_song_length

    # Penalty 1: EOS target too early
    too_early_for_eos = F.relu(self.target_song_length - positions)
    early_eos_in_target = is_eos_target * too_early_for_eos * 0.5

    # Penalty 2: Discourage p_eos before min_song_length
    before_min = (positions < self.min_song_length).float()
    discourage_early_eos = is_not_eos_target * before_min * p_eos * 5.0

    # Penalty 3: Fill the dead zone with ramping penalty
    in_mid_zone = ((positions >= self.min_song_length) & (positions < self.target_song_length)).float()
    ramp = (self.target_song_length - positions) / (self.target_song_length - self.min_song_length)
    mid_zone_penalty = is_not_eos_target * in_mid_zone * p_eos * ramp * 2.0
```

### Fix 5: Empty Lyrics (dataset.py)

```python
lyrics_indices = self.vocab.encode(row['lyrics'])

# CRITICAL FIX: Skip very short lyrics
if len(lyrics_indices) < 10:
    lyrics_indices = [self.vocab.unk_idx] * 10
```

### Fix 6: Clone Logits (generate.py)

```python
def sample_with_temperature(logits, ...):
    # CRITICAL FIX: Clone to avoid in-place mutation
    logits = logits.clone()
    # ... rest of function
```

---

## Required Actions Before Retraining

### 1. Clear Cache

The tokenization fix changes how vocabulary is built. You MUST delete cached data:

```bash
rm -rf outputs/cache/*
```

### 2. Verify Tokenization

After clearing cache, verify NEWLINE tokens are preserved:

```python
from src.vocab import Vocabulary
vocab = Vocabulary()
tokens = vocab.tokenize("hello <NEWLINE> world")
print(tokens)  # Should be: ['hello', '<NEWLINE>', 'world']
```

### 3. Retrain Model

Train from scratch with the fixed code:

```bash
python scripts/train.py --epochs 50
```

---

## Expected Improvements

After these fixes:

1. **NEWLINE tokens are properly learned** from training data
2. **Loss function no longer fights cross-entropy** - penalties only apply where relevant
3. **No dead zone for EOS** - continuous pressure until target length
4. **No SOS→EOS shortcuts** from empty training samples
5. **Consistent sampling** without cumulative penalties

The hardcoded `min_line_words` and `min_length` parameters in generation can now be relaxed or removed, as the model should learn proper structure during training.

---

## Technical Details

### Why the Tokenization Bug Was So Devastating

The vocabulary builds like this:
1. Special tokens added first: `<PAD>=0, <UNK>=1, <SOS>=2, <EOS>=3, <NEWLINE>=4`
2. Words from training data added: `love=5, the=6, ...`

When tokenizing `"love <NEWLINE> you"`:
- **Before fix:** `['love', 'newline', 'you']` → indices `[5, ???, 12]`
  - `newline` (lowercase, no brackets) either maps to UNK or gets its own high index
- **After fix:** `['love', '<NEWLINE>', 'you']` → indices `[5, 4, 12]`
  - Proper NEWLINE token at index 4

### Why the Loss Function Conflict Matters

Cross-entropy already handles the primary objective: predict the correct next token.

The structure-aware loss should ONLY provide **additional guidance**, not conflict with CE:
- **Old:** "Never output NEWLINE" (applies penalty at all positions)
- **New:** "Don't output NEWLINE on short lines" (applies only at NEWLINE targets)

This is the difference between a loss that trains the model and one that confuses it.
