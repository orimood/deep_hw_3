"""Configuration settings for the lyrics generation project."""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Data files
TRAIN_CSV = DATA_DIR / "lyrics_train_set.csv"
TEST_CSV = DATA_DIR / "lyrics_test_set.csv"
MIDI_DIR = DATA_DIR / "midi_files"

# Word2Vec settings
WORD2VEC_DIM = 300
WORD2VEC_MODEL = "word2vec-google-news-300"  # Pre-trained model from gensim

# MIDI feature settings
MIDI_FEATURE_DIM = 128  # Dimension of MIDI feature representation
MIDI_PIANO_ROLL_FS = 100  # Sampling frequency for piano roll
MIDI_MAX_LENGTH = 1000  # Maximum frames to consider from MIDI

# Model settings
HIDDEN_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.3
BIDIRECTIONAL = False

# Training settings
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
TEACHER_FORCING_RATIO = 0.5
GRADIENT_CLIP = 5.0
VALIDATION_SPLIT = 0.1

# Generation settings
MAX_LYRICS_LENGTH = 200  # Maximum words to generate
MAX_WORDS_PER_LINE = 10  # Approximate words per line
TEMPERATURE = 0.8  # Sampling temperature (lower = more deterministic)
TOP_K = 50  # Top-k sampling parameter

# Special tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
NEWLINE_TOKEN = "<NEWLINE>"

# Device
DEVICE = "cuda"  # Will be updated at runtime based on availability

# Random seed for reproducibility
SEED = 42
