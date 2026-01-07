#!/usr/bin/env python3
"""
Main evaluation script for lyrics generation models.

Usage:
    python run_evaluation.py --approach global
    python run_evaluation.py --approach attention
    python run_evaluation.py --approach both
"""

import argparse
import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config
from src.dataset import LyricsDataset, get_test_dataset, get_dataloaders
from src.model import create_model
from src.train import load_checkpoint
from src.evaluate import evaluate_model, compare_models


def main():
    parser = argparse.ArgumentParser(description='Evaluate lyrics generation model')
    parser.add_argument(
        '--approach',
        type=str,
        choices=['global', 'attention', 'both'],
        default='both',
        help='Which model approach to evaluate'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to specific checkpoint (default: best model)'
    )
    args = parser.parse_args()

    print("=" * 60)
    print("LYRICS GENERATION - EVALUATION")
    print("=" * 60)
    print(f"Approach: {args.approach}")
    print(f"Device: {config.DEVICE}")
    print("=" * 60)

    # Load training dataset to get vocab and embeddings
    print("\nLoading data...")
    _, _, train_dataset = get_dataloaders(use_cache=True)

    vocab = train_dataset.vocab
    embedding_matrix = train_dataset.embedding_matrix

    # Load test dataset
    print("Loading test dataset...")
    test_dataset = get_test_dataset(train_dataset)
    print(f"Test songs: {len(test_dataset)}")

    # Evaluate based on approach
    if args.approach == 'both':
        # Load both models
        print("\nLoading Global model...")
        global_model = create_model(
            approach='global',
            vocab_size=len(vocab),
            embedding_matrix=embedding_matrix,
            device=config.DEVICE
        )
        global_checkpoint = config.MODELS_DIR / "lyrics_global_best.pt"
        if global_checkpoint.exists():
            global_model = load_checkpoint(global_model, global_checkpoint, config.DEVICE)
        else:
            print(f"WARNING: No checkpoint found at {global_checkpoint}")

        print("\nLoading Attention model...")
        attention_model = create_model(
            approach='attention',
            vocab_size=len(vocab),
            embedding_matrix=embedding_matrix,
            device=config.DEVICE
        )
        attention_checkpoint = config.MODELS_DIR / "lyrics_attention_best.pt"
        if attention_checkpoint.exists():
            attention_model = load_checkpoint(attention_model, attention_checkpoint, config.DEVICE)
        else:
            print(f"WARNING: No checkpoint found at {attention_checkpoint}")

        # Compare both models
        compare_models(
            global_model=global_model,
            attention_model=attention_model,
            test_dataset=test_dataset,
            vocab=vocab,
            device=config.DEVICE
        )

    else:
        # Load single model
        print(f"\nLoading {args.approach} model...")
        model = create_model(
            approach=args.approach,
            vocab_size=len(vocab),
            embedding_matrix=embedding_matrix,
            device=config.DEVICE
        )

        if args.checkpoint:
            checkpoint_path = Path(args.checkpoint)
        else:
            checkpoint_path = config.MODELS_DIR / f"lyrics_{args.approach}_best.pt"

        if checkpoint_path.exists():
            model = load_checkpoint(model, checkpoint_path, config.DEVICE)
        else:
            print(f"WARNING: No checkpoint found at {checkpoint_path}")

        # Evaluate
        evaluate_model(
            model=model,
            test_dataset=test_dataset,
            vocab=vocab,
            device=config.DEVICE,
            is_attention_model=(args.approach == 'attention'),
            model_name=f"lyrics_{args.approach}"
        )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE!")
    print("=" * 60)
    print(f"\nResults saved in: {config.RESULTS_DIR}")


if __name__ == '__main__':
    main()
