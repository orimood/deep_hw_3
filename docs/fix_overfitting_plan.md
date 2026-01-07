# Plan: Fix Overfitting and Short Song Generation

## Problem Summary
1. **Overfitting**: Validation loss increases from epoch 0 while training loss decreases
2. **Short songs**: Generated output stops too early (~14 words avg vs target 100-150)
3. **First line quality**: First line is good (model learns short-term patterns)

## Root Causes
- **Overfitting**: No weight decay, moderate dropout (0.3), no data augmentation
- **Short songs**: EOS token predicted too early, no minimum length enforcement during generation

---

## Changes

### 1. Add Weight Decay to Optimizer
**File:** `src/train.py` (line ~156)

```python
# BEFORE
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# AFTER
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
```

### 2. Increase Dropout
**File:** `src/config.py` (line 51)

```python
# BEFORE
DROPOUT = 0.3

# AFTER
DROPOUT = 0.5
```

### 3. Reduce Early Stopping Patience
**File:** `src/config.py` (line 66)

```python
# BEFORE
EARLY_STOPPING_PATIENCE = 10

# AFTER
EARLY_STOPPING_PATIENCE = 5
```

### 4. Add Minimum Length Enforcement in Generation
**File:** `src/generate.py` (in both `generate_lyrics_global` and `generate_lyrics_attention`)

Add parameter and logic to skip EOS token before minimum length:

```python
def generate_lyrics_global(model, vocab, midi_features, start_words,
                          max_length=200, temperature=0.8, top_k=50,
                          min_length=100):  # NEW PARAMETER
    ...
    # In the generation loop, after sampling:
    if next_idx == vocab.eos_idx:
        if len(generated_tokens) < min_length:
            # Skip EOS, sample again excluding EOS
            logits[vocab.eos_idx] = float('-inf')
            next_idx = sample_with_temperature(logits, temperature, top_k)
        else:
            break
```

### 5. Adjust Structure Loss for Consistent 100-150 Words
**File:** `src/config.py` (lines 100-110)

```python
STRUCTURE_LOSS = {
    'target_line_length': 6.0,
    'line_length_sigma': 3.5,
    'lambda_line': 0.3,              # Reduced from 0.5 (less aggressive)
    'target_song_length': 120.0,     # Reduced from 150 (target middle of 100-150)
    'length_scale': 30.0,            # Reduced from 50 (tighter sigmoid)
    'lambda_length': 0.1,            # Reduced from 0.2 (less aggressive)
    'newline_weight': 3.0,           # Keep as is
    'min_line_length': 3,            # Keep as is
    'min_song_length': 80,           # Increased from 50
}
```

---

## Files to Modify
1. `src/config.py` - Dropout, early stopping, structure loss
2. `src/train.py` - Add weight decay to optimizer
3. `src/generate.py` - Add min_length enforcement

## Expected Outcome
- **Overfitting**: Validation loss should track closer to training loss
- **Length**: Generated songs should consistently be 100-150 words
- **Quality**: First line quality preserved (short-term patterns unaffected)

## Training Command
```bash
python scripts/train.py --model attention --epochs 50
```
