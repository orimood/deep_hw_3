"""LSTM models for lyrics generation with melody conditioning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

from . import config


class LyricsLSTMGlobal(nn.Module):
    """
    Approach 1: Global Melody Conditioning

    The MIDI features are extracted as a single global vector and concatenated
    with the word embedding at each timestep.

    Architecture:
    - Word embedding (300-dim from Word2Vec)
    - MIDI features (128-dim global vector)
    - Concatenate: [word_emb || midi_feat] = 428-dim
    - LSTM layers
    - Output projection to vocabulary
    """

    def __init__(self,
                 vocab_size: int,
                 embedding_matrix: np.ndarray,
                 hidden_dim: int = config.HIDDEN_DIM,
                 num_layers: int = config.NUM_LAYERS,
                 dropout: float = config.DROPOUT,
                 midi_dim: int = config.MIDI_FEATURE_DIM,
                 freeze_embeddings: bool = True):

        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding_dim = embedding_matrix.shape[1]
        self.midi_dim = midi_dim

        # Word embedding layer (initialized with Word2Vec)
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        # MIDI feature projection
        self.midi_projection = nn.Sequential(
            nn.Linear(midi_dim, midi_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # LSTM input: word embedding + projected MIDI features
        lstm_input_dim = self.embedding_dim + midi_dim

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self,
                input_seq: torch.Tensor,
                midi_features: torch.Tensor,
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple]:
        """
        Forward pass.

        Args:
            input_seq: Input word indices [batch, seq_len]
            midi_features: Global MIDI features [batch, midi_dim]
            hidden: Previous hidden state (h, c)
            lengths: Sequence lengths for packing

        Returns:
            output: Logits for next word [batch, seq_len, vocab_size]
            hidden: Updated hidden state
        """
        batch_size, seq_len = input_seq.shape

        # Get word embeddings
        embeddings = self.embedding(input_seq)  # [batch, seq_len, emb_dim]

        # Project and expand MIDI features
        midi_proj = self.midi_projection(midi_features)  # [batch, midi_dim]
        midi_expanded = midi_proj.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, midi_dim]

        # Concatenate word embeddings with MIDI features
        lstm_input = torch.cat([embeddings, midi_expanded], dim=2)  # [batch, seq_len, emb_dim + midi_dim]

        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size, input_seq.device)

        # LSTM forward pass
        lstm_out, hidden = self.lstm(lstm_input, hidden)  # [batch, seq_len, hidden_dim]

        # Output projection
        output = self.dropout(lstm_out)
        output = self.fc_out(output)  # [batch, seq_len, vocab_size]

        return output, hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state."""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h0, c0)


class MelodyAttention(nn.Module):
    """
    Attention mechanism over temporal MIDI features.

    Computes attention weights based on the LSTM hidden state
    and returns a weighted combination of MIDI frame features.
    """

    def __init__(self, hidden_dim: int, midi_frame_dim: int):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.midi_frame_dim = midi_frame_dim

        # Attention layers
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_m = nn.Linear(midi_frame_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self,
                hidden: torch.Tensor,
                midi_sequence: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention over MIDI sequence.

        Args:
            hidden: LSTM hidden state [batch, hidden_dim]
            midi_sequence: Temporal MIDI features [batch, num_frames, midi_frame_dim]
            mask: Optional mask for MIDI frames [batch, num_frames]

        Returns:
            context: Weighted MIDI context [batch, midi_frame_dim]
            attention_weights: Attention weights [batch, num_frames]
        """
        batch_size, num_frames, _ = midi_sequence.shape

        # Project hidden state and MIDI sequence
        hidden_proj = self.W_h(hidden).unsqueeze(1)  # [batch, 1, hidden_dim]
        midi_proj = self.W_m(midi_sequence)  # [batch, num_frames, hidden_dim]

        # Compute attention scores
        energy = torch.tanh(hidden_proj + midi_proj)  # [batch, num_frames, hidden_dim]
        scores = self.v(energy).squeeze(-1)  # [batch, num_frames]

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)  # [batch, num_frames]

        # Compute context vector
        context = torch.bmm(attention_weights.unsqueeze(1), midi_sequence)  # [batch, 1, midi_frame_dim]
        context = context.squeeze(1)  # [batch, midi_frame_dim]

        return context, attention_weights


class LyricsLSTMAttention(nn.Module):
    """
    Approach 2: Temporal Melody Alignment with Attention

    Instead of using global MIDI features, this model extracts temporal
    MIDI features (a sequence of feature vectors) and uses an attention
    mechanism to attend to relevant melody positions at each word.

    Architecture:
    - Word embedding (300-dim from Word2Vec)
    - Temporal MIDI features (num_frames x frame_dim)
    - Attention mechanism: attend to MIDI based on hidden state
    - LSTM with attention context
    - Output projection to vocabulary
    """

    def __init__(self,
                 vocab_size: int,
                 embedding_matrix: np.ndarray,
                 hidden_dim: int = config.HIDDEN_DIM,
                 num_layers: int = config.NUM_LAYERS,
                 dropout: float = config.DROPOUT,
                 midi_frame_dim: int = config.MIDI_FEATURE_DIM // 4,
                 num_midi_frames: int = 50,
                 freeze_embeddings: bool = True):

        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding_dim = embedding_matrix.shape[1]
        self.midi_frame_dim = midi_frame_dim
        self.num_midi_frames = num_midi_frames

        # Word embedding layer (initialized with Word2Vec)
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        # MIDI frame projection
        self.midi_projection = nn.Sequential(
            nn.Linear(midi_frame_dim, midi_frame_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Attention mechanism
        self.attention = MelodyAttention(hidden_dim, midi_frame_dim)

        # LSTM input: word embedding + attention context
        lstm_input_dim = self.embedding_dim + midi_frame_dim

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self,
                input_seq: torch.Tensor,
                midi_sequence: torch.Tensor,
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple, torch.Tensor]:
        """
        Forward pass with attention.

        Args:
            input_seq: Input word indices [batch, seq_len]
            midi_sequence: Temporal MIDI features [batch, num_frames, frame_dim]
            hidden: Previous hidden state (h, c)
            lengths: Sequence lengths

        Returns:
            output: Logits for next word [batch, seq_len, vocab_size]
            hidden: Updated hidden state
            attention_weights: Attention weights for each step [batch, seq_len, num_frames]
        """
        batch_size, seq_len = input_seq.shape

        # Get word embeddings
        embeddings = self.embedding(input_seq)  # [batch, seq_len, emb_dim]

        # Project MIDI sequence
        midi_proj = self.midi_projection(midi_sequence)  # [batch, num_frames, frame_dim]

        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size, input_seq.device)

        # Process sequence step by step with attention
        outputs = []
        all_attention_weights = []

        for t in range(seq_len):
            # Get current word embedding
            word_emb = embeddings[:, t, :]  # [batch, emb_dim]

            # Get current hidden state for attention
            h_t = hidden[0][-1]  # [batch, hidden_dim] (last layer)

            # Compute attention over MIDI sequence
            context, attn_weights = self.attention(h_t, midi_proj)  # [batch, frame_dim]
            all_attention_weights.append(attn_weights)

            # Concatenate word embedding with attention context
            lstm_input = torch.cat([word_emb, context], dim=1)  # [batch, emb_dim + frame_dim]
            lstm_input = lstm_input.unsqueeze(1)  # [batch, 1, emb_dim + frame_dim]

            # LSTM step
            lstm_out, hidden = self.lstm(lstm_input, hidden)  # [batch, 1, hidden_dim]
            outputs.append(lstm_out)

        # Stack outputs
        lstm_out = torch.cat(outputs, dim=1)  # [batch, seq_len, hidden_dim]
        attention_weights = torch.stack(all_attention_weights, dim=1)  # [batch, seq_len, num_frames]

        # Output projection
        output = self.dropout(lstm_out)
        output = self.fc_out(output)  # [batch, seq_len, vocab_size]

        return output, hidden, attention_weights

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state."""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h0, c0)


def create_model(approach: str,
                 vocab_size: int,
                 embedding_matrix: np.ndarray,
                 device: torch.device) -> nn.Module:
    """
    Factory function to create the appropriate model.

    Args:
        approach: "global" or "attention"
        vocab_size: Size of vocabulary
        embedding_matrix: Pre-trained Word2Vec embeddings
        device: Device to put model on

    Returns:
        Model instance
    """
    if approach == "global":
        model = LyricsLSTMGlobal(
            vocab_size=vocab_size,
            embedding_matrix=embedding_matrix,
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT,
            midi_dim=config.MIDI_FEATURE_DIM
        )
    elif approach == "attention":
        model = LyricsLSTMAttention(
            vocab_size=vocab_size,
            embedding_matrix=embedding_matrix,
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT,
            midi_frame_dim=config.MIDI_FEATURE_DIM // 4,
            num_midi_frames=50
        )
    else:
        raise ValueError(f"Unknown approach: {approach}. Use 'global' or 'attention'.")

    return model.to(device)
