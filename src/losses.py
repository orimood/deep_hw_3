"""
Structure-aware loss functions for lyrics generation.

Teaches the model proper lyric structure through training:
1. Cross-entropy with boosted NEWLINE weight
2. Line length regularization
3. Song length guidance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

from . import config
from .vocab import Vocabulary


class StructureAwareLoss(nn.Module):
    """
    Multi-component loss function for lyrics generation.

    L_total = L_ce + λ_line * L_line + λ_length * L_length

    Components:
    1. Weighted Cross-Entropy: Standard CE with boosted NEWLINE token weight
    2. Line Length Regularization: Penalizes too-short or too-long lines
    3. Song Length Guidance: Sigmoid pressure to encourage appropriate length
    """

    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        newline_idx: int,
        eos_idx: int,
        target_line_length: float = config.STRUCTURE_LOSS['target_line_length'],
        line_length_sigma: float = config.STRUCTURE_LOSS['line_length_sigma'],
        lambda_line: float = config.STRUCTURE_LOSS['lambda_line'],
        target_song_length: float = config.STRUCTURE_LOSS['target_song_length'],
        length_scale: float = config.STRUCTURE_LOSS['length_scale'],
        lambda_length: float = config.STRUCTURE_LOSS['lambda_length'],
        newline_weight: float = config.STRUCTURE_LOSS['newline_weight'],
        min_line_length: int = config.STRUCTURE_LOSS['min_line_length'],
        min_song_length: int = config.STRUCTURE_LOSS['min_song_length']
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
        self.min_line_length = min_line_length

        # Song length parameters
        self.target_song_length = target_song_length
        self.length_scale = length_scale
        self.lambda_length = lambda_length
        self.min_song_length = min_song_length

        # Create class weights for CE loss
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
        Compute structure-aware loss.

        Args:
            logits: Model output [batch, seq_len, vocab_size]
            targets: Target indices [batch, seq_len]
            return_components: If True, also return loss breakdown

        Returns:
            Total loss (optionally with component breakdown)
        """
        batch_size, seq_len, vocab_size = logits.shape
        device = logits.device

        # Ensure weights are on correct device
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

        # 3. Song Length Guidance
        length_loss = self._compute_length_guidance_loss(logits, targets)

        # Combine losses
        total_loss = ce_loss + self.lambda_line * line_loss + self.lambda_length * length_loss

        if return_components:
            return total_loss, {
                'ce_loss': ce_loss.item(),
                'line_loss': line_loss.item(),
                'length_loss': length_loss.item(),
                'total_loss': total_loss.item()
            }

        return total_loss

    def _compute_line_length_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Line length regularization.

        Encourages NEWLINE at appropriate positions:
        - Penalizes low NEWLINE probability when line is too long
        - Penalizes high NEWLINE probability when line is too short
        - HARD penalty for very short lines (< min_line_length)
        """
        batch_size, seq_len, _ = logits.shape
        device = logits.device

        # Get NEWLINE probabilities
        probs = F.softmax(logits, dim=-1)
        p_newline = probs[:, :, self.newline_idx]  # [batch, seq_len]

        # Compute position in current line for each token
        line_positions = self._compute_line_positions(targets)  # [batch, seq_len]

        # Mask for non-padding positions
        mask = (targets != self.pad_idx).float()

        # Late penalty: encourage NEWLINE when past target length
        distance_past_target = F.relu(line_positions - self.target_line_length)
        late_penalty = distance_past_target * (1 - p_newline) / self.line_length_sigma

        # Early penalty: discourage NEWLINE when line is too short
        # Using EXPONENTIAL penalty for stronger effect
        early_positions = F.relu(self.min_line_length - line_positions)
        early_penalty = (torch.exp(early_positions) - 1) * p_newline * 5.0

        # HARD penalty for very short lines (< min_line_length words)
        # Using 100x multiplier with exponential scaling
        very_short = (line_positions < self.min_line_length).float()
        shortness_factor = (self.min_line_length - line_positions).clamp(min=0)
        hard_penalty = very_short * p_newline * 100.0 * (1 + shortness_factor)  # 100x penalty

        # Combined penalty
        penalty = late_penalty + early_penalty + hard_penalty

        # Apply mask and compute mean
        loss = (penalty * mask).sum() / (mask.sum() + 1e-8)

        return loss

    def _compute_line_positions(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute position within line for each token.

        Returns count of words since last NEWLINE (or start of sequence).
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

        return positions

    def _compute_length_guidance_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Song length guidance.

        Uses sigmoid pressure to encourage EOS prediction at appropriate song length.
        Also penalizes early EOS (before min_song_length).
        """
        batch_size, seq_len, _ = logits.shape
        device = logits.device

        # Get EOS probabilities
        probs = F.softmax(logits, dim=-1)
        p_eos = probs[:, :, self.eos_idx]  # [batch, seq_len]

        # Position in sequence
        positions = torch.arange(seq_len, device=device, dtype=torch.float)
        positions = positions.unsqueeze(0).expand(batch_size, -1)

        # Mask for non-padding
        mask = (targets != self.pad_idx).float()

        # Sigmoid pressure: increases after target length
        eos_pressure = torch.sigmoid((positions - self.target_song_length) / self.length_scale)

        # Penalty: high pressure but low EOS probability (encourage EOS after target)
        late_penalty = eos_pressure * (1 - p_eos)

        # HARD penalty for early EOS (before min_song_length)
        # Using EXPONENTIAL penalty with 100x multiplier
        too_short = (positions < self.min_song_length).float()
        # Exponential: penalty grows exponentially as we get further from min length
        distance_from_min = (self.min_song_length - positions).clamp(min=0) / 10.0  # Scale down for exp
        early_eos_penalty = too_short * p_eos * 100.0 * torch.exp(distance_from_min)  # 100x with exp

        # Combined penalty
        penalty = late_penalty + early_eos_penalty

        # Apply mask and compute mean
        loss = (penalty * mask).sum() / (mask.sum() + 1e-8)

        return loss


def create_loss_function(vocab: Vocabulary, device: torch.device = config.DEVICE) -> StructureAwareLoss:
    """
    Factory function to create structure-aware loss from vocabulary.

    Args:
        vocab: Vocabulary object with word2idx mapping
        device: Device to put loss on

    Returns:
        StructureAwareLoss instance
    """
    return StructureAwareLoss(
        vocab_size=len(vocab),
        pad_idx=vocab.pad_idx,
        newline_idx=vocab.newline_idx,
        eos_idx=vocab.eos_idx,
        target_line_length=config.STRUCTURE_LOSS['target_line_length'],
        line_length_sigma=config.STRUCTURE_LOSS['line_length_sigma'],
        lambda_line=config.STRUCTURE_LOSS['lambda_line'],
        target_song_length=config.STRUCTURE_LOSS['target_song_length'],
        length_scale=config.STRUCTURE_LOSS['length_scale'],
        lambda_length=config.STRUCTURE_LOSS['lambda_length'],
        newline_weight=config.STRUCTURE_LOSS['newline_weight'],
        min_line_length=config.STRUCTURE_LOSS['min_line_length'],
        min_song_length=config.STRUCTURE_LOSS['min_song_length']
    ).to(device)
