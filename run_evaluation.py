#!/usr/bin/env python3
"""
Evaluation script for lyrics generation models.

This script generates lyrics for the test set using both trained models
with multiple starting words.

Usage:
    python run_evaluation.py
    python run_evaluation.py --start_words "love,the,dream"
"""

import argparse
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src import config
from src.data_loader import LyricsDataset, get_dataloaders, get_test_dataset
from src.models import create_model
from src.train import load_checkpoint
from src.generate import evaluate_test_set, generate_lyrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate lyrics generation models")
    parser.add_argument(
        "--start_words",
        type=str,
        default="love,the,i",
        help="Comma-separated list of starting words"
    )
    parser.add_argument(
        "--global_checkpoint",
        type=str,
        default=None,
        help="Path to global model checkpoint"
    )
    parser.add_argument(
        "--attention_checkpoint",
        type=str,
        default=None,
        help="Path to attention model checkpoint"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=config.TEMPERATURE,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=config.TOP_K,
        help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/mps/cpu)"
    )

    args = parser.parse_args()

    # Parse starting words
    start_words = [w.strip() for w in args.start_words.split(',')]
    print(f"Starting words: {start_words}")

    # Set device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Default checkpoint paths
    if args.global_checkpoint is None:
        args.global_checkpoint = config.MODELS_DIR / "lyrics_lstm_global_best.pt"
    if args.attention_checkpoint is None:
        args.attention_checkpoint = config.MODELS_DIR / "lyrics_lstm_attention_best.pt"

    # Check if checkpoints exist
    if not Path(args.global_checkpoint).exists():
        print(f"Error: Global model checkpoint not found at {args.global_checkpoint}")
        print("Please train the models first using: python run_training.py")
        sys.exit(1)

    if not Path(args.attention_checkpoint).exists():
        print(f"Error: Attention model checkpoint not found at {args.attention_checkpoint}")
        print("Please train the models first using: python run_training.py")
        sys.exit(1)

    # Load training dataset to get vocabulary and embedding matrix
    print("\n" + "="*60)
    print("Loading data...")
    print("="*60)

    train_loader, val_loader, train_dataset = get_dataloaders()
    vocab = train_dataset.vocab
    embedding_matrix = train_dataset.embedding_matrix
    vocab_size = len(vocab)

    print(f"Vocabulary size: {vocab_size}")

    # Load test dataset
    test_dataset = get_test_dataset(train_dataset)
    print(f"Test songs: {len(test_dataset)}")

    # Create and load global model
    print("\n" + "="*60)
    print("Loading Global Melody Conditioning model...")
    print("="*60)

    global_model = create_model(
        approach="global",
        vocab_size=vocab_size,
        embedding_matrix=embedding_matrix,
        device=device
    )
    global_model = load_checkpoint(global_model, str(args.global_checkpoint), device)
    global_model.eval()

    # Create and load attention model
    print("\n" + "="*60)
    print("Loading Attention model...")
    print("="*60)

    attention_model = create_model(
        approach="attention",
        vocab_size=vocab_size,
        embedding_matrix=embedding_matrix,
        device=device
    )
    attention_model = load_checkpoint(attention_model, str(args.attention_checkpoint), device)
    attention_model.eval()

    # Generate lyrics for test set
    print("\n" + "="*60)
    print("Generating lyrics for test set...")
    print("="*60)

    # Update config with command line args
    results = evaluate_test_set(
        global_model=global_model,
        attention_model=attention_model,
        test_dataset=test_dataset,
        vocab=vocab,
        device=device,
        start_words=start_words,
        output_dir=config.OUTPUTS_DIR
    )

    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60)
    print(f"\nGenerated lyrics saved to: {config.OUTPUTS_DIR / 'generated_lyrics.txt'}")


if __name__ == "__main__":
    main()
