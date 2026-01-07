"""Structure-aware loss functions for lyrics generation.

This module provides loss functions that teach the model proper lyric structure
through training, rather than relying on hardcoded rules during generation.

Components:
1. Cross-entropy loss with boosted weight for <NEWLINE> token
2. Line length regularization (encourages appropriate line breaks)
3. Total length guidance (encourages appropriate song length)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

from . import config


class StructureAwareLoss(nn.Module):
    """
    Multi-component loss function that teaches lyric structure through training.

    L_total = L_ce + λ_line * L_line + λ_length * L_length

    Components:
    1. Cross-entropy loss (primary task) with boosted <NEWLINE> weight
    2. Line length regularization - penalizes missing/premature newlines
    3. Total length guidance - sigmoid pressure for appropriate song length
    """

    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        newline_idx: int,
        eos_idx: int,
        # Line length parameters (from training data: median=6, std=3.7)
        target_line_length: float = 6.0,
        line_length_sigma: float = 3.5,
        lambda_line: float = 0.1,
        # Total length parameters (from training data: median=235, std=146)
        target_song_length: float = 235.0,
        length_scale: float = 75.0,
        lambda_length: float = 0.05,
        # Newline weighting
        newline_weight: float = 2.0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.newline_idx = newline_idx
        self.eos_idx = eos_idx

        # Line length parameters
        self.target_line_length = target_line_length
        self.line_length_sigma = line_length_sigma
        self.lambda_line = lambda_line

        # Total length parameters
        self.target_song_length = target_song_length
        self.length_scale = length_scale
        self.lambda_length = lambda_length

        # Newline weighting
        self.newline_weight = newline_weight

        # Create class weights for cross-entropy
        weights = torch.ones(vocab_size)
        weights[newline_idx] = newline_weight
        self.register_buffer('class_weights', weights)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute the structure-aware loss.

        Args:
            logits: Model output logits [batch, seq_len, vocab_size]
            targets: Target token indices [batch, seq_len]
            return_components: If True, return individual loss components

        Returns:
            Total loss (and optionally a dict of loss components)
        """
        batch_size, seq_len, vocab_size = logits.shape
        device = logits.device

        # Move weights to device if needed
        weights = self.class_weights.to(device)

        # 1. Weighted Cross-Entropy Loss
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)

        ce_loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            weight=weights,
            ignore_index=self.pad_idx,
            reduction='mean'
        )

        # 2. Line Length Regularization
        line_loss = self._compute_line_length_loss(logits, targets)

        # 3. Total Length Guidance
        length_loss = self._compute_length_guidance_loss(logits, targets)

        # Combine losses
        total_loss = (
            ce_loss +
            self.lambda_line * line_loss +
            self.lambda_length * length_loss
        )

        if return_components:
            components = {
                'ce_loss': ce_loss.item(),
                'line_loss': line_loss.item(),
                'length_loss': length_loss.item(),
                'total_loss': total_loss.item()
            }
            return total_loss, components

        return total_loss

    def _compute_line_length_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute line length regularization loss.

        Encourages model to predict <NEWLINE> at appropriate positions:
        - Penalizes low <NEWLINE> probability when line is getting long (>target)
        - Penalizes high <NEWLINE> probability when line is too short (<3 words)
        """
        batch_size, seq_len, vocab_size = logits.shape
        device = logits.device

        # Get softmax probabilities for newline token
        probs = F.softmax(logits, dim=-1)
        p_newline = probs[:, :, self.newline_idx]  # [batch, seq_len]

        # Build position-in-line tensor
        line_positions = self._compute_line_positions(targets)  # [batch, seq_len]

        # Create mask for non-padding positions
        mask = (targets != self.pad_idx).float()

        # For positions past the target line length, penalize low newline probability
        # The further past target, the higher the penalty
        distance_past_target = F.relu(line_positions - self.target_line_length)
        late_penalty = distance_past_target * (1 - p_newline) / self.line_length_sigma

        # For very early positions (<4 words), penalize high newline probability
        # Quadratic penalty to strongly discourage short lines
        early_positions = F.relu(4 - line_positions)
        early_penalty = (early_positions ** 2) * p_newline

        # Combine penalties
        penalty = late_penalty + early_penalty

        # Apply mask and compute mean
        loss = (penalty * mask).sum() / (mask.sum() + 1e-8)

        return loss

    def _compute_line_positions(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute position within line for each token.

        Returns tensor where each position contains the count of words
        since the last <NEWLINE> token (or start of sequence).
        """
        batch_size, seq_len = targets.shape
        device = targets.device

        positions = torch.zeros_like(targets, dtype=torch.float, device=device)

        for b in range(batch_size):
            count = 0
            for t in range(seq_len):
                if targets[b, t] == self.newline_idx:
                    positions[b, t] = count
                    count = 0
                elif targets[b, t] != self.pad_idx:
                    count += 1
                    positions[b, t] = count
                # For padding, position stays 0

        return positions

    def _compute_length_guidance_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss that encourages EOS prediction at appropriate song length.

        Uses sigmoid pressure that increases as we go past target song length.
        """
        batch_size, seq_len, vocab_size = logits.shape
        device = logits.device

        # Get softmax probabilities for EOS token
        probs = F.softmax(logits, dim=-1)
        p_eos = probs[:, :, self.eos_idx]  # [batch, seq_len]

        # Create position tensor (global position in sequence)
        positions = torch.arange(seq_len, device=device, dtype=torch.float)
        positions = positions.unsqueeze(0).expand(batch_size, -1)

        # Create mask for non-padding positions
        mask = (targets != self.pad_idx).float()

        # Sigmoid pressure that increases after target length
        # Encourages higher EOS probability as we go past target length
        eos_pressure = torch.sigmoid(
            (positions - self.target_song_length) / self.length_scale
        )

        # Penalty: high pressure but low EOS probability
        penalty = eos_pressure * (1 - p_eos)

        # Apply mask and compute mean
        loss = (penalty * mask).sum() / (mask.sum() + 1e-8)

        return loss


def create_structure_loss(vocab, device: torch.device) -> StructureAwareLoss:
    """
    Factory function to create structure-aware loss from vocabulary.

    Args:
        vocab: Vocabulary object with word2idx mapping
        device: Device to put loss on

    Returns:
        StructureAwareLoss instance
    """
    # Get hyperparameters from config
    loss_config = getattr(config, 'STRUCTURE_LOSS', {})

    return StructureAwareLoss(
        vocab_size=len(vocab),
        pad_idx=vocab.word2idx[config.PAD_TOKEN],
        newline_idx=vocab.word2idx[config.NEWLINE_TOKEN],
        eos_idx=vocab.word2idx[config.EOS_TOKEN],
        target_line_length=loss_config.get('target_line_length', 6.0),
        line_length_sigma=loss_config.get('line_length_sigma', 3.5),
        lambda_line=loss_config.get('lambda_line', 0.1),
        target_song_length=loss_config.get('target_song_length', 235.0),
        length_scale=loss_config.get('length_scale', 75.0),
        lambda_length=loss_config.get('lambda_length', 0.05),
        newline_weight=loss_config.get('newline_weight', 2.0),
    ).to(device)
