"""
LSTM models for lyrics generation with melody conditioning.

Two approaches as required by the assignment:
1. Global Melody Conditioning - concatenate global MIDI features
2. Temporal Melody with Attention - attend to MIDI frames dynamically

Architecture patterns from:
- FloydHub: dropout on embeddings, weight initialization
- DebuggerCafe: batch_first=True, hidden state management
- TensorFlow: state management for generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

import config


class LyricsLSTMGlobal(nn.Module):
    """
    Approach 1: Global Melody Conditioning

    MIDI features are extracted as a single global vector and concatenated
    with the word embedding at each timestep.

    Architecture:
        Word Embedding (300-dim Word2Vec)
        + MIDI Global Features (128-dim)
        → Concatenate (428-dim)
        → LSTM (2 layers, 512 hidden)
        → Dropout
        → Linear (vocab_size)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_matrix: np.ndarray,
        hidden_dim: int = config.HIDDEN_DIM,
        num_layers: int = config.NUM_LAYERS,
        dropout: float = config.DROPOUT,
        midi_dim: int = config.MIDI_GLOBAL_DIM,
        freeze_embeddings: bool = config.FREEZE_EMBEDDINGS
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding_dim = embedding_matrix.shape[1]
        self.midi_dim = midi_dim

        # Word embedding (initialized with Word2Vec)
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        # Dropout layer (FloydHub pattern: dropout on embeddings AND output)
        self.dropout = nn.Dropout(dropout)

        # MIDI feature projection
        self.midi_projection = nn.Sequential(
            nn.Linear(midi_dim, midi_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # LSTM: input = word_embedding + midi_features
        lstm_input_dim = self.embedding_dim + midi_dim

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,  # DebuggerCafe pattern
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # Output projection
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

        # Initialize weights (FloydHub pattern)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights uniformly (FloydHub pattern)."""
        initrange = 0.1
        self.fc_out.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()

    def forward(
        self,
        input_seq: torch.Tensor,
        midi_features: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            input_seq: Input word indices [batch, seq_len]
            midi_features: Global MIDI features [batch, midi_dim]
            hidden: Previous hidden state (h, c)

        Returns:
            output: Logits [batch, seq_len, vocab_size]
            hidden: Updated hidden state (h, c)
        """
        batch_size, seq_len = input_seq.shape

        # Get word embeddings with dropout (FloydHub pattern)
        embeddings = self.dropout(self.embedding(input_seq))  # [batch, seq_len, emb_dim]

        # Project MIDI features and expand to sequence length
        midi_proj = self.midi_projection(midi_features)  # [batch, midi_dim]
        midi_expanded = midi_proj.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, midi_dim]

        # Concatenate embeddings with MIDI features
        lstm_input = torch.cat([embeddings, midi_expanded], dim=2)  # [batch, seq_len, emb_dim + midi_dim]

        # Initialize hidden if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size, input_seq.device)

        # LSTM forward
        lstm_out, hidden = self.lstm(lstm_input, hidden)  # [batch, seq_len, hidden_dim]

        # Output projection with dropout
        output = self.dropout(lstm_out)
        output = self.fc_out(output)  # [batch, seq_len, vocab_size]

        return output, hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state with zeros."""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h0, c0)


class MelodyAttention(nn.Module):
    """
    Attention mechanism over temporal MIDI features.

    Computes attention weights based on LSTM hidden state
    and returns weighted combination of MIDI frame features.
    """

    def __init__(self, hidden_dim: int, midi_frame_dim: int):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.midi_frame_dim = midi_frame_dim

        # Attention projections (Bahdanau-style)
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_m = nn.Linear(midi_frame_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        midi_sequence: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention over MIDI sequence.

        Args:
            hidden: LSTM hidden state [batch, hidden_dim]
            midi_sequence: Temporal MIDI features [batch, num_frames, midi_frame_dim]
            mask: Optional mask [batch, num_frames]

        Returns:
            context: Weighted MIDI context [batch, midi_frame_dim]
            attention_weights: Attention weights [batch, num_frames]
        """
        # Project hidden state and MIDI sequence
        hidden_proj = self.W_h(hidden).unsqueeze(1)  # [batch, 1, hidden_dim]
        midi_proj = self.W_m(midi_sequence)  # [batch, num_frames, hidden_dim]

        # Compute attention scores
        energy = torch.tanh(hidden_proj + midi_proj)  # [batch, num_frames, hidden_dim]
        scores = self.v(energy).squeeze(-1)  # [batch, num_frames]

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)  # [batch, num_frames]

        # Compute weighted context
        context = torch.bmm(attention_weights.unsqueeze(1), midi_sequence)  # [batch, 1, midi_frame_dim]
        context = context.squeeze(1)  # [batch, midi_frame_dim]

        return context, attention_weights


class LyricsLSTMAttention(nn.Module):
    """
    Approach 2: Temporal Melody with Attention

    Uses temporal MIDI features (sequence of frame vectors) with attention
    mechanism to dynamically focus on relevant melody parts at each word.

    Architecture:
        Word Embedding (300-dim Word2Vec)
        + Attention Context (32-dim from MIDI temporal)
        → Concatenate (332-dim)
        → LSTM (2 layers, 512 hidden)
        → Dropout
        → Linear (vocab_size)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_matrix: np.ndarray,
        hidden_dim: int = config.HIDDEN_DIM,
        num_layers: int = config.NUM_LAYERS,
        dropout: float = config.DROPOUT,
        midi_frame_dim: int = config.MIDI_FRAME_DIM,
        num_midi_frames: int = config.MIDI_TEMPORAL_FRAMES,
        freeze_embeddings: bool = config.FREEZE_EMBEDDINGS
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding_dim = embedding_matrix.shape[1]
        self.midi_frame_dim = midi_frame_dim
        self.num_midi_frames = num_midi_frames

        # Word embedding (initialized with Word2Vec)
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # MIDI frame projection
        self.midi_projection = nn.Sequential(
            nn.Linear(midi_frame_dim, midi_frame_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Attention mechanism
        self.attention = MelodyAttention(hidden_dim, midi_frame_dim)

        # LSTM: input = word_embedding + attention_context
        lstm_input_dim = self.embedding_dim + midi_frame_dim

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # Output projection
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights uniformly."""
        initrange = 0.1
        self.fc_out.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()

    def forward(
        self,
        input_seq: torch.Tensor,
        midi_sequence: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Forward pass with attention.

        Args:
            input_seq: Input word indices [batch, seq_len]
            midi_sequence: Temporal MIDI features [batch, num_frames, frame_dim]
            hidden: Previous hidden state (h, c)

        Returns:
            output: Logits [batch, seq_len, vocab_size]
            hidden: Updated hidden state (h, c)
            attention_weights: Attention weights [batch, seq_len, num_frames]
        """
        batch_size, seq_len = input_seq.shape

        # Get word embeddings with dropout
        embeddings = self.dropout(self.embedding(input_seq))  # [batch, seq_len, emb_dim]

        # Project MIDI sequence
        midi_proj = self.midi_projection(midi_sequence)  # [batch, num_frames, frame_dim]

        # Initialize hidden if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size, input_seq.device)

        # Process sequence step by step with attention
        outputs = []
        all_attention_weights = []

        for t in range(seq_len):
            # Current word embedding
            word_emb = embeddings[:, t, :]  # [batch, emb_dim]

            # Get hidden state for attention query (last layer)
            h_t = hidden[0][-1]  # [batch, hidden_dim]

            # Compute attention over MIDI sequence
            context, attn_weights = self.attention(h_t, midi_proj)  # [batch, frame_dim]
            all_attention_weights.append(attn_weights)

            # Concatenate word embedding with attention context
            lstm_input = torch.cat([word_emb, context], dim=1)  # [batch, emb_dim + frame_dim]
            lstm_input = lstm_input.unsqueeze(1)  # [batch, 1, emb_dim + frame_dim]

            # LSTM step
            lstm_out, hidden = self.lstm(lstm_input, hidden)  # [batch, 1, hidden_dim]
            outputs.append(lstm_out)

        # Stack outputs and attention weights
        lstm_out = torch.cat(outputs, dim=1)  # [batch, seq_len, hidden_dim]
        attention_weights = torch.stack(all_attention_weights, dim=1)  # [batch, seq_len, num_frames]

        # Output projection with dropout
        output = self.dropout(lstm_out)
        output = self.fc_out(output)  # [batch, seq_len, vocab_size]

        return output, hidden, attention_weights

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state with zeros."""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h0, c0)


def create_model(
    approach: str,
    vocab_size: int,
    embedding_matrix: np.ndarray,
    device: torch.device = config.DEVICE
) -> nn.Module:
    """
    Factory function to create the appropriate model.

    Args:
        approach: "global" or "attention"
        vocab_size: Size of vocabulary
        embedding_matrix: Pre-trained Word2Vec embeddings
        device: Device to put model on

    Returns:
        Model instance on specified device
    """
    if approach == "global":
        model = LyricsLSTMGlobal(
            vocab_size=vocab_size,
            embedding_matrix=embedding_matrix,
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT,
            midi_dim=config.MIDI_GLOBAL_DIM
        )
    elif approach == "attention":
        model = LyricsLSTMAttention(
            vocab_size=vocab_size,
            embedding_matrix=embedding_matrix,
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT,
            midi_frame_dim=config.MIDI_FRAME_DIM,
            num_midi_frames=config.MIDI_TEMPORAL_FRAMES
        )
    else:
        raise ValueError(f"Unknown approach: {approach}. Use 'global' or 'attention'.")

    model = model.to(device)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{approach.upper()} Model:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    return model
