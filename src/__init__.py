"""
Lyrics Generation Pipeline

A PyTorch implementation for generating song lyrics conditioned on MIDI melodies.

Two approaches implemented:
1. Global Melody Conditioning - Uses global MIDI features concatenated with embeddings
2. Temporal Melody with Attention - Attends over temporal MIDI frames dynamically

Usage:
    # Training
    python run_training.py --approach both

    # Evaluation
    python run_evaluation.py --approach both

    # Interactive generation
    python run_generate.py --approach global --start-words "love is"
"""

from .config import *
from .vocab import Vocabulary, build_vocab_from_texts
from .midi_features import extract_global_features, extract_temporal_features
from .dataset import LyricsDataset, get_dataloaders, get_test_dataset
from .model import LyricsLSTMGlobal, LyricsLSTMAttention, create_model
from .losses import StructureAwareLoss, create_loss_function
from .train import train_model, load_checkpoint
from .generate import (
    generate_lyrics_global,
    generate_lyrics_attention,
    sample_with_temperature
)
from .evaluate import evaluate_model, compare_models
