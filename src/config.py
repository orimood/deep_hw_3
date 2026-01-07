"""
Configuration settings for the lyrics generation pipeline.
All hyperparameters consolidated from multiple sources:
- FloydHub word-language-model
- DebuggerCafe LSTM tutorial
- TensorFlow text generation
- MachineLearningMastery
"""

from pathlib import Path
import torch

# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
SRC_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CACHE_DIR = OUTPUTS_DIR / "cache"
MODELS_DIR = OUTPUTS_DIR / "models"
RESULTS_DIR = OUTPUTS_DIR / "results"
RUNS_DIR = OUTPUTS_DIR / "runs"

# Data files
TRAIN_CSV = DATA_DIR / "lyrics_train_set.csv"
TEST_CSV = DATA_DIR / "lyrics_test_set.csv"
MIDI_DIR = DATA_DIR / "midi_files"

# =============================================================================
# Word2Vec Settings (Assignment requirement: 300-dim)
# =============================================================================
EMBEDDING_DIM = 300
WORD2VEC_MODEL = "word2vec-google-news-300"
FREEZE_EMBEDDINGS = True  # Freeze pre-trained embeddings

# =============================================================================
# MIDI Feature Settings
# =============================================================================
MIDI_GLOBAL_DIM = 128       # Global feature vector dimension
MIDI_TEMPORAL_FRAMES = 50   # Number of temporal frames
MIDI_FRAME_DIM = 32         # Features per frame
MIDI_PIANO_ROLL_FS = 100    # Piano roll sampling frequency

# =============================================================================
# Model Architecture
# Source: FloydHub uses 200-1500, we use 512 for balance
# =============================================================================
HIDDEN_DIM = 512            # LSTM hidden units
NUM_LAYERS = 2              # Number of LSTM layers
DROPOUT = 0.3               # Dropout rate (FloydHub uses 0.2-0.65)
BIDIRECTIONAL = False       # Unidirectional for generation

# =============================================================================
# Training Settings
# Source: Combined from all references
# =============================================================================
BATCH_SIZE = 32             # DebuggerCafe: 32, FloydHub: 20
LEARNING_RATE = 0.001       # Adam default
NUM_EPOCHS = 50             # With early stopping
GRADIENT_CLIP = 5.0         # FloydHub uses 0.25, we use 5.0
TEACHER_FORCING_RATIO = 0.5 # 50% teacher forcing
VALIDATION_SPLIT = 0.1      # 10% validation

# Early stopping (from FloydHub pattern)
EARLY_STOPPING_PATIENCE = 10
LR_DECAY_FACTOR = 0.5       # Reduce LR by half on plateau
LR_DECAY_PATIENCE = 5       # Wait 5 epochs before reducing

# =============================================================================
# Sequence Settings
# =============================================================================
MAX_SEQ_LENGTH = 200        # Maximum sequence length
MIN_WORD_FREQUENCY = 1      # Minimum word frequency (set higher to filter rare words)

# =============================================================================
# Generation Settings
# Source: TensorFlow tutorial temperature guidance
# =============================================================================
TEMPERATURE = 0.8           # 0.7-1.0 sweet spot for creativity
TOP_K = 50                  # Top-k sampling
MAX_GENERATION_LENGTH = 200  # Max words to generate

# =============================================================================
# Special Tokens
# =============================================================================
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
NEWLINE_TOKEN = "<NEWLINE>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN, NEWLINE_TOKEN]

# =============================================================================
# Structure-Aware Loss Settings
# Derived from training data analysis
# =============================================================================
STRUCTURE_LOSS = {
    'target_line_length': 6.0,      # Median words per line
    'line_length_sigma': 3.5,       # Std dev
    'lambda_line': 0.1,             # Line length loss weight
    'target_song_length': 235.0,    # Median total words
    'length_scale': 75.0,           # Sigmoid scale
    'lambda_length': 0.05,          # Length loss weight
    'newline_weight': 2.0,          # Boost for NEWLINE token in CE
}

# =============================================================================
# Device Configuration
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# Random Seed
# =============================================================================
SEED = 42

# =============================================================================
# Test Configuration
# =============================================================================
TEST_START_WORDS = ["love", "the", "i"]  # 3 starting words as required
