#!/usr/bin/env python3
"""
Main training script for lyrics generation models.

Usage:
    python run_training.py --approach global
    python run_training.py --approach attention
    python run_training.py --approach both
"""

import argparse
import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config
from src.dataset import get_dataloaders
from src.model import create_model
from src.train import train_model


def main():
    parser = argparse.ArgumentParser(description='Train lyrics generation model')
    parser.add_argument(
        '--approach',
        type=str,
        choices=['global', 'attention', 'both'],
        default='both',
        help='Which melody integration approach to train'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=config.NUM_EPOCHS,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=config.BATCH_SIZE,
        help='Batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=config.LEARNING_RATE,
        help='Learning rate'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable caching of vocab, embeddings, MIDI features'
    )
    args = parser.parse_args()

    # Print configuration
    print("=" * 60)
    print("LYRICS GENERATION - TRAINING")
    print("=" * 60)
    print(f"Approach: {args.approach}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {config.DEVICE}")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    train_loader, val_loader, train_dataset = get_dataloaders(
        batch_size=args.batch_size,
        use_cache=not args.no_cache
    )

    vocab = train_dataset.vocab
    embedding_matrix = train_dataset.embedding_matrix

    print(f"\nVocabulary size: {len(vocab)}")
    print(f"Embedding shape: {embedding_matrix.shape}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # Train models based on approach
    approaches_to_train = []
    if args.approach == 'both':
        approaches_to_train = ['global', 'attention']
    else:
        approaches_to_train = [args.approach]

    for approach in approaches_to_train:
        print("\n" + "=" * 60)
        print(f"TRAINING: {approach.upper()} MELODY MODEL")
        print("=" * 60)

        # Create model
        model = create_model(
            approach=approach,
            vocab_size=len(vocab),
            embedding_matrix=embedding_matrix,
            device=config.DEVICE
        )

        # Train
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            vocab=vocab,
            device=config.DEVICE,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            model_name=f"lyrics_{approach}",
            is_attention_model=(approach == 'attention')
        )

        print(f"\n{approach.upper()} model training complete!")
        print(f"Best model saved to: {config.MODELS_DIR / f'lyrics_{approach}_best.pt'}")

    print("\n" + "=" * 60)
    print("ALL TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nModels saved in: {config.MODELS_DIR}")
    print(f"TensorBoard logs in: {config.RUNS_DIR}")
    print(f"\nTo view TensorBoard: tensorboard --logdir {config.RUNS_DIR}")


if __name__ == '__main__':
    main()
