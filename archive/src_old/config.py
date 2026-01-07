"""Configuration settings for the lyrics generation project."""

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

# Structure-aware loss settings (values derived from training data analysis)
# Training data: 599 songs, 23,860 lines
# Line length: median=6, mean=6.6, std=3.7, 75% in range 5-8 words
# Song length: median=235, mean=261, std=146, 75% in range 172-305 words
STRUCTURE_LOSS = {
    'target_line_length': 6.0,    # Median words per line from data
    'line_length_sigma': 3.5,     # Std dev from data
    'lambda_line': 0.5,           # Weight for line length loss (aggressive)
    'target_song_length': 235.0,  # Median total words from data
    'length_scale': 75.0,         # Scale for length sigmoid (~half std)
    'lambda_length': 0.2,         # Weight for length loss (aggressive)
    'newline_weight': 1.0,        # CE weight boost for <NEWLINE> token (reduced)
}
