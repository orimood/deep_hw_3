# Lyrics Generation Using RNNs with Melody Conditioning

Deep Learning Assignment 3 - Ben-Gurion University

Generate song lyrics conditioned on MIDI melody information using LSTM networks with two different melody integration approaches.

## Project Structure

```
.
├── src/
│   ├── config.py          # Hyperparameters and settings
│   ├── models.py          # LSTM models (Global & Attention)
│   ├── data_loader.py     # Dataset and vocabulary
│   ├── midi_features.py   # MIDI feature extraction
│   ├── train.py           # Training loop
│   └── generate.py        # Lyrics generation
├── data/
│   ├── lyrics_train_set.csv   # 599 training songs
│   ├── lyrics_test_set.csv    # 4 test songs
│   └── midi_files/            # 627 MIDI files
├── models/                # Saved model checkpoints
├── outputs/               # Generated lyrics
├── run_training.py        # Training entry point
├── run_evaluation.py      # Generation entry point
└── requirements.txt       # Dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- PyTorch 2.0+
- pretty_midi
- gensim (for Word2Vec)
- tensorboard

## Usage

### Training

Train both model variants:
```bash
python run_training.py --approach both
```

Train specific approach:
```bash
python run_training.py --approach global      # Global conditioning only
python run_training.py --approach attention   # Attention model only
```

Monitor training with TensorBoard:
```bash
tensorboard --logdir runs/
```

### Generating Lyrics

Generate lyrics for test melodies:
```bash
python run_evaluation.py
```

With custom starting words:
```bash
python run_evaluation.py --start_words "love,the,i"
```

## Model Architectures

### Approach 1: Global Melody Conditioning

Extracts a single 128-dimensional global feature vector from the MIDI file and concatenates it with word embeddings at each timestep.

- Word embeddings: 300-dim (Word2Vec)
- MIDI features: 128-dim global vector
- LSTM: 2 layers, 512 hidden units
- Dropout: 0.3

### Approach 2: Temporal Melody with Attention

Uses a sequence of temporal MIDI features with an attention mechanism that allows the model to focus on different parts of the melody during generation.

- Word embeddings: 300-dim (Word2Vec)
- Temporal MIDI features: 50 frames x 32 dimensions
- Attention: Query from LSTM hidden state, keys/values from MIDI sequence
- LSTM: 2 layers, 512 hidden units
- Dropout: 0.3

## MIDI Feature Extraction

Global features (128-dim):
- Tempo statistics (7 features)
- Piano roll activity (32 features)
- Instrument distribution (18 features)
- Note statistics (20 features)
- Chroma/pitch class (12 features)

Temporal features (50 x 32):
- Time-aligned piano roll representations
- Used only by the attention model

## Generation

- **Sampling**: Top-k (k=50) with temperature scaling (T=0.8)
- **Max length**: 200 words
- **Line breaks**: Max 10 words per line, uses `<NEWLINE>` token

## Configuration

Key hyperparameters in `src/config.py`:

| Parameter | Value |
|-----------|-------|
| Learning rate | 0.001 |
| Batch size | 32 |
| Epochs | 50 |
| Hidden dim | 512 |
| LSTM layers | 2 |
| Dropout | 0.3 |
| Teacher forcing | 0.5 |
| Train/Val split | 90/10 |
