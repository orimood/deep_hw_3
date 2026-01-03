"""Training loop for lyrics generation models."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import numpy as np
import random
import math
from pathlib import Path
from typing import Dict, Tuple, Optional
import time

from . import config
from .models import LyricsLSTMGlobal, LyricsLSTMAttention


def set_seed(seed: int = config.SEED):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model: nn.Module,
                train_loader,
                optimizer: optim.Optimizer,
                criterion: nn.Module,
                device: torch.device,
                gradient_clip: float = config.GRADIENT_CLIP,
                is_attention_model: bool = False) -> float:
    """
    Train for one epoch.

    Args:
        model: The model to train
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        gradient_clip: Gradient clipping value
        is_attention_model: Whether model uses attention

    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(train_loader, desc="Training", leave=False):
        inputs, targets, midi_features, midi_temporal, lengths = batch

        inputs = inputs.to(device)
        targets = targets.to(device)
        midi_features = midi_features.to(device)
        midi_temporal = midi_temporal.to(device)

        optimizer.zero_grad()

        # Forward pass
        if is_attention_model:
            # Use real temporal MIDI features for attention model
            outputs, _, attention_weights = model(inputs, midi_temporal)
        else:
            # Use global MIDI features for global model
            outputs, _ = model(inputs, midi_features)

        # Compute loss
        # outputs: [batch, seq_len, vocab_size]
        # targets: [batch, seq_len]
        outputs_flat = outputs.view(-1, outputs.size(-1))
        targets_flat = targets.view(-1)

        # Ignore padding in loss
        loss = criterion(outputs_flat, targets_flat)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if gradient_clip > 0:
            clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def validate(model: nn.Module,
             val_loader,
             criterion: nn.Module,
             device: torch.device,
             is_attention_model: bool = False) -> float:
    """
    Validate the model.

    Args:
        model: The model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        is_attention_model: Whether model uses attention

    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            inputs, targets, midi_features, midi_temporal, lengths = batch

            inputs = inputs.to(device)
            targets = targets.to(device)
            midi_features = midi_features.to(device)
            midi_temporal = midi_temporal.to(device)

            # Forward pass
            if is_attention_model:
                # Use real temporal MIDI features
                outputs, _, _ = model(inputs, midi_temporal)
            else:
                outputs, _ = model(inputs, midi_features)

            # Compute loss
            outputs_flat = outputs.view(-1, outputs.size(-1))
            targets_flat = targets.view(-1)

            loss = criterion(outputs_flat, targets_flat)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def train_model(model: nn.Module,
                train_loader,
                val_loader,
                device: torch.device,
                num_epochs: int = config.NUM_EPOCHS,
                learning_rate: float = config.LEARNING_RATE,
                model_name: str = "lyrics_lstm",
                is_attention_model: bool = False) -> Dict[str, list]:
    """
    Full training loop.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        model_name: Name for saving checkpoints
        is_attention_model: Whether model uses attention

    Returns:
        Dictionary with training history
    """
    set_seed()

    # Setup
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Loss function - ignore padding token (index 0)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # TensorBoard writer
    log_dir = config.PROJECT_ROOT / "runs" / f"{model_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=str(log_dir))

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }

    best_val_loss = float('inf')
    save_dir = config.MODELS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training {model_name} for {num_epochs} epochs...")
    print(f"TensorBoard logs: {log_dir}")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            is_attention_model=is_attention_model
        )

        # Validate
        val_loss = validate(model, val_loader, criterion, device,
                           is_attention_model=is_attention_model)

        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Log to history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(current_lr)

        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Perplexity/train', math.exp(train_loss), epoch)
        writer.add_scalar('Perplexity/validation', math.exp(val_loss), epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)

        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  LR:         {current_lr:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss
            }
            torch.save(checkpoint, save_dir / f"{model_name}_best.pt")
            print(f"  Saved best model (val_loss: {val_loss:.4f})")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss
            }
            torch.save(checkpoint, save_dir / f"{model_name}_epoch{epoch+1}.pt")

    writer.close()
    print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")

    return history


def load_checkpoint(model: nn.Module,
                    checkpoint_path: str,
                    device: torch.device) -> nn.Module:
    """Load a model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.4f}")
    return model
