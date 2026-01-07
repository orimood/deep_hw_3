#!/usr/bin/env python3
"""
Main training script for lyrics generation models.

This script trains both model approaches:
1. Global Melody Conditioning (LyricsLSTMGlobal)
2. Attention over Melody (LyricsLSTMAttention)

Usage:
    python run_training.py --approach global
    python run_training.py --approach attention
    python run_training.py --approach both
"""

import argparse
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src import config
from src.data_loader import get_dataloaders
from src.models import create_model
from src.train import train_model, set_seed


def main():
    parser = argparse.ArgumentParser(description="Train lyrics generation models")
    parser.add_argument(
        "--approach",
        type=str,
        choices=["global", "attention", "both"],
        default="both",
        help="Which approach to train: 'global', 'attention', or 'both'"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.NUM_EPOCHS,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config.BATCH_SIZE,
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=config.LEARNING_RATE,
        help="Learning rate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/mps/cpu). Auto-detected if not specified."
    )

    args = parser.parse_args()

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

    # Set random seed
    set_seed(config.SEED)

    # Create output directories
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n" + "="*60)
    print("Loading data...")
    print("="*60)

    train_loader, val_loader, train_dataset = get_dataloaders(
        batch_size=args.batch_size,
        val_split=config.VALIDATION_SPLIT
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Vocabulary size: {len(train_dataset.vocab)}")

    # Get embedding matrix
    embedding_matrix = train_dataset.embedding_matrix
    vocab_size = len(train_dataset.vocab)

    approaches_to_train = []
    if args.approach == "both":
        approaches_to_train = ["global", "attention"]
    else:
        approaches_to_train = [args.approach]

    # Train each approach
    for approach in approaches_to_train:
        print("\n" + "="*60)
        print(f"Training {approach.upper()} model...")
        print("="*60)

        # Create model
        model = create_model(
            approach=approach,
            vocab_size=vocab_size,
            embedding_matrix=embedding_matrix,
            device=device
        )

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        # Train
        is_attention = (approach == "attention")
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            vocab=train_dataset.vocab,
            device=device,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            model_name=f"lyrics_lstm_{approach}",
            is_attention_model=is_attention
        )

        print(f"\n{approach.upper()} model training complete!")
        print(f"Best model saved to: {config.MODELS_DIR}/lyrics_lstm_{approach}_best.pt")

    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print(f"\nTo view training progress, run:")
    print(f"  tensorboard --logdir={config.PROJECT_ROOT / 'runs'}")


if __name__ == "__main__":
    main()
