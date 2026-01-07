"""
Training loop for lyrics generation models.

Includes patterns from:
- FloydHub: gradient clipping, learning rate decay, early stopping
- DebuggerCafe: training loop structure
"""

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
from typing import Dict, Optional
import time

from . import config
from .model import LyricsLSTMGlobal, LyricsLSTMAttention
from .losses import create_loss_function


def set_seed(seed: int = config.SEED):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: nn.Module,
    train_loader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    gradient_clip: float = config.GRADIENT_CLIP,
    is_attention_model: bool = False
) -> float:
    """
    Train for one epoch.

    Pattern from DebuggerCafe with gradient clipping from FloydHub.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(train_loader, desc="Training", leave=False)

    for batch in progress_bar:
        inputs, targets, midi_global, midi_temporal, lengths = batch

        inputs = inputs.to(device)
        targets = targets.to(device)
        midi_global = midi_global.to(device)
        midi_temporal = midi_temporal.to(device)

        optimizer.zero_grad()

        # Forward pass
        if is_attention_model:
            outputs, _, _ = model(inputs, midi_temporal)
        else:
            outputs, _ = model(inputs, midi_global)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping (FloydHub pattern)
        if gradient_clip > 0:
            clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / num_batches


def validate(
    model: nn.Module,
    val_loader,
    criterion: nn.Module,
    device: torch.device,
    is_attention_model: bool = False
) -> float:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            inputs, targets, midi_global, midi_temporal, lengths = batch

            inputs = inputs.to(device)
            targets = targets.to(device)
            midi_global = midi_global.to(device)
            midi_temporal = midi_temporal.to(device)

            # Forward pass
            if is_attention_model:
                outputs, _, _ = model(inputs, midi_temporal)
            else:
                outputs, _ = model(inputs, midi_global)

            # Compute loss
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    vocab,
    device: torch.device,
    num_epochs: int = config.NUM_EPOCHS,
    learning_rate: float = config.LEARNING_RATE,
    model_name: str = "lyrics_lstm",
    is_attention_model: bool = False
) -> Dict[str, list]:
    """
    Full training loop with early stopping and learning rate decay.

    Patterns from FloydHub:
    - Learning rate decay on plateau
    - Early stopping
    - Gradient clipping
    - Model checkpointing
    """
    set_seed()

    # Create directories
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # Setup optimizer and scheduler (added weight_decay for L2 regularization)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Learning rate scheduler (FloydHub pattern: reduce on plateau)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.LR_DECAY_FACTOR,
        patience=config.LR_DECAY_PATIENCE
    )

    # Loss function
    criterion = create_loss_function(vocab, device)

    # TensorBoard writer
    log_dir = config.RUNS_DIR / f"{model_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=str(log_dir))

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': [],
        'perplexity_train': [],
        'perplexity_val': []
    }

    # Early stopping variables (FloydHub pattern)
    best_val_loss = float('inf')
    early_stopping_counter = 0

    print(f"\nTraining {model_name} for up to {num_epochs} epochs...")
    print(f"Early stopping patience: {config.EARLY_STOPPING_PATIENCE}")
    print(f"TensorBoard logs: {log_dir}")
    print(f"Device: {device}")
    print("-" * 60)

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            is_attention_model=is_attention_model
        )

        # Validate
        val_loss = validate(
            model, val_loader, criterion, device,
            is_attention_model=is_attention_model
        )

        # Update learning rate scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Calculate perplexity
        train_ppl = math.exp(min(train_loss, 100))  # Cap to prevent overflow
        val_ppl = math.exp(min(val_loss, 100))

        # Log to history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(current_lr)
        history['perplexity_train'].append(train_ppl)
        history['perplexity_val'].append(val_ppl)

        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Perplexity/train', train_ppl, epoch)
        writer.add_scalar('Perplexity/validation', val_ppl, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)

        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch + 1}/{num_epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f} | Perplexity: {train_ppl:.2f}")
        print(f"  Val Loss:   {val_loss:.4f} | Perplexity: {val_ppl:.2f}")
        print(f"  LR: {current_lr:.6f}")

        # Check for best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0

            # Save best model
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'history': history
            }
            save_path = config.MODELS_DIR / f"{model_name}_best.pt"
            torch.save(checkpoint, save_path)
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
        else:
            early_stopping_counter += 1
            print(f"  No improvement ({early_stopping_counter}/{config.EARLY_STOPPING_PATIENCE})")

        # Early stopping check (FloydHub pattern)
        if early_stopping_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss
            }
            torch.save(checkpoint, config.MODELS_DIR / f"{model_name}_epoch{epoch+1}.pt")

        print()

    writer.close()

    print("=" * 60)
    print(f"Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {config.MODELS_DIR / f'{model_name}_best.pt'}")

    return history


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device
) -> nn.Module:
    """Load a model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Val Loss: {checkpoint['val_loss']:.4f}")

    return model
