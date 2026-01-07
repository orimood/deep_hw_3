#!/usr/bin/env python3
"""
Training script for GRU-based lyrics generation models.

Usage:
    python run_gru_training.py --approach global
    python run_gru_training.py --approach attention
    python run_gru_training.py --approach both
"""

import argparse
import torch
import sys
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config
from src.data_loader import get_dataloaders
from src.losses import create_structure_loss
from gru_models import create_gru_model

import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random
import numpy as np


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_gru_model(model: nn.Module,
                    train_loader,
                    val_loader,
                    vocab,
                    device: torch.device,
                    num_epochs: int = config.NUM_EPOCHS,
                    learning_rate: float = config.LEARNING_RATE,
                    model_name: str = "lyrics_gru",
                    is_attention_model: bool = False) -> dict:
    """Train a GRU model."""

    # Create output directories
    models_dir = Path(__file__).parent / "models"
    runs_dir = Path(__file__).parent / "runs"
    models_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Setup loss and optimizer
    criterion = create_structure_loss(vocab, device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # TensorBoard writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(runs_dir / f"{model_name}_{timestamp}")
    print(f"TensorBoard logs: {runs_dir / f'{model_name}_{timestamp}'}")

    # Training history
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')

    print(f"Training {model_name} for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            inputs, targets, midi_features, midi_temporal, lengths = batch

            inputs = inputs.to(device)
            targets = targets.to(device)
            midi_features = midi_features.to(device)
            midi_temporal = midi_temporal.to(device)

            optimizer.zero_grad()

            # Forward pass
            if is_attention_model:
                outputs, _, _ = model(inputs, midi_temporal)
            else:
                outputs, _ = model(inputs, midi_features)

            # Compute loss
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                inputs, targets, midi_features, midi_temporal, lengths = batch

                inputs = inputs.to(device)
                targets = targets.to(device)
                midi_features = midi_features.to(device)
                midi_temporal = midi_temporal.to(device)

                if is_attention_model:
                    outputs, _, _ = model(inputs, midi_temporal)
                else:
                    outputs, _ = model(inputs, midi_features)

                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Update scheduler
        scheduler.step(val_loss)

        # Log metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  LR:         {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
            }, models_dir / f"{model_name}_best.pt")
            print(f"  Saved best model (val_loss: {val_loss:.4f})")

        # Save periodic checkpoints
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
            }, models_dir / f"{model_name}_epoch{epoch+1}.pt")

        print()

    writer.close()
    print(f"Training complete. Best validation loss: {best_val_loss:.4f}")

    return history


def main():
    parser = argparse.ArgumentParser(description="Train GRU lyrics generation models")
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
        print(f"Training GRU {approach.upper()} model...")
        print("="*60)

        # Create model
        model = create_gru_model(
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
        history = train_gru_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            vocab=train_dataset.vocab,
            device=device,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            model_name=f"lyrics_gru_{approach}",
            is_attention_model=is_attention
        )

        print(f"\nGRU {approach.upper()} model training complete!")
        print(f"Best model saved to: gru_experiment/models/lyrics_gru_{approach}_best.pt")

    print("\n" + "="*60)
    print("GRU Training complete!")
    print("="*60)
    print(f"\nTo view training progress, run:")
    print(f"  tensorboard --logdir=gru_experiment/runs")


if __name__ == "__main__":
    main()
