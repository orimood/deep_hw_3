"""
Text generation with temperature sampling and top-k filtering.

Pattern from TensorFlow tutorial: use multinomial sampling, NOT argmax.
This produces more diverse and creative outputs.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path

from . import config
from .vocab import Vocabulary
from .model import LyricsLSTMGlobal, LyricsLSTMAttention


def sample_with_temperature(
    logits: torch.Tensor,
    temperature: float = config.TEMPERATURE,
    top_k: int = config.TOP_K,
    repetition_penalty_tokens: List[int] = None,
    repetition_penalty: float = 1.0
) -> int:
    """
    Sample from logits with temperature scaling and top-k filtering.

    Pattern from TensorFlow tutorial: use multinomial, not argmax.

    Args:
        logits: Raw logits [vocab_size]
        temperature: Controls randomness (lower = more deterministic)
        top_k: Only sample from top k tokens
        repetition_penalty_tokens: Token indices to penalize
        repetition_penalty: Penalty factor (>1.0 reduces probability)

    Returns:
        Sampled token index
    """
    # Apply repetition penalty to specific tokens
    if repetition_penalty_tokens and repetition_penalty > 1.0:
        for token_idx in repetition_penalty_tokens:
            if token_idx < logits.size(-1):
                logits[token_idx] = logits[token_idx] / repetition_penalty

    # Apply temperature
    logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        # Get top-k values and indices
        top_k_values, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))

        # Create mask for non-top-k tokens
        logits_filtered = torch.full_like(logits, float('-inf'))
        logits_filtered.scatter_(0, top_k_indices, top_k_values)
        logits = logits_filtered

    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)

    # Sample using multinomial (TensorFlow pattern: NOT argmax)
    sampled_idx = torch.multinomial(probs, num_samples=1).item()

    return sampled_idx


def generate_lyrics_global(
    model: LyricsLSTMGlobal,
    vocab: Vocabulary,
    midi_features: np.ndarray,
    start_words: List[str],
    max_length: int = config.MAX_GENERATION_LENGTH,
    temperature: float = config.TEMPERATURE,
    top_k: int = config.TOP_K,
    device: torch.device = config.DEVICE,
    min_length: int = 100,
    min_line_words: int = 4
) -> Tuple[str, List[str]]:
    """
    Generate lyrics using global melody conditioning model.

    Args:
        model: Trained LyricsLSTMGlobal model
        vocab: Vocabulary object
        midi_features: Global MIDI features [midi_dim]
        start_words: List of starting words
        max_length: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k filtering parameter
        device: Device to run on
        min_length: Minimum tokens before allowing EOS (enforces longer output)
        min_line_words: Minimum words before allowing NEWLINE (prevents short lines)

    Returns:
        generated_text: Full generated lyrics string
        token_list: List of generated tokens
    """
    model.eval()

    # Prepare MIDI features
    midi_tensor = torch.tensor(midi_features, dtype=torch.float32).unsqueeze(0).to(device)

    # Get NEWLINE token indices
    newline_indices = []
    for token in [config.NEWLINE_TOKEN, '<newline>', 'newline']:
        if token in vocab.word2idx:
            newline_indices.append(vocab.word2idx[token])

    # Encode starting words
    tokens = [vocab.sos_idx]
    for word in start_words:
        word_lower = word.lower()
        if word_lower in vocab.word2idx:
            tokens.append(vocab.word2idx[word_lower])
        else:
            tokens.append(vocab.unk_idx)

    generated_tokens = list(start_words)
    hidden = None
    words_since_newline = len(start_words)  # Track words on current line

    with torch.no_grad():
        # Process start sequence to get initial hidden state
        input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
        outputs, hidden = model(input_tensor, midi_tensor, hidden)

        # Get logits for last position
        logits = outputs[0, -1, :]

        # Generate tokens one by one
        for _ in range(max_length - len(start_words)):
            # Sample next token
            next_idx = sample_with_temperature(logits, temperature, top_k)

            # Enforce minimum line length - skip NEWLINE if line too short
            if next_idx in newline_indices and words_since_newline < min_line_words:
                # Re-sample excluding NEWLINE tokens
                logits_no_newline = logits.clone()
                for nl_idx in newline_indices:
                    logits_no_newline[nl_idx] = float('-inf')
                next_idx = sample_with_temperature(logits_no_newline, temperature, top_k)

            # Check for EOS - skip if under min_length
            if next_idx == vocab.eos_idx:
                if len(generated_tokens) >= min_length:
                    break
                else:
                    # Force re-sample excluding EOS
                    logits[vocab.eos_idx] = float('-inf')
                    next_idx = sample_with_temperature(logits, temperature, top_k)

            # Convert to word and add to generated
            word = vocab.idx2word.get(next_idx, config.UNK_TOKEN)
            generated_tokens.append(word)

            # Track words since newline
            if next_idx in newline_indices or word.lower() in ['<newline>', 'newline']:
                words_since_newline = 0
            else:
                words_since_newline += 1

            # Prepare next input
            next_input = torch.tensor([[next_idx]], dtype=torch.long).to(device)

            # Forward pass for single token
            outputs, hidden = model(next_input, midi_tensor, hidden)
            logits = outputs[0, -1, :]

    # Format output with proper newlines
    text = _format_lyrics(generated_tokens)

    return text, generated_tokens


def generate_lyrics_attention(
    model: LyricsLSTMAttention,
    vocab: Vocabulary,
    midi_temporal: np.ndarray,
    start_words: List[str],
    max_length: int = config.MAX_GENERATION_LENGTH,
    temperature: float = config.TEMPERATURE,
    top_k: int = config.TOP_K,
    device: torch.device = config.DEVICE,
    min_length: int = 100,
    newline_penalty: float = 2.5,
    min_line_words: int = 4
) -> Tuple[str, List[str], torch.Tensor]:
    """
    Generate lyrics using attention-based melody conditioning model.

    Args:
        model: Trained LyricsLSTMAttention model
        vocab: Vocabulary object
        midi_temporal: Temporal MIDI features [num_frames, frame_dim]
        start_words: List of starting words
        max_length: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k filtering parameter
        device: Device to run on
        min_length: Minimum tokens before allowing EOS (enforces longer output)
        newline_penalty: Penalty for consecutive NEWLINEs (prevents degeneration)
        min_line_words: Minimum words before allowing NEWLINE (prevents short lines)

    Returns:
        generated_text: Full generated lyrics string
        token_list: List of generated tokens
        attention_weights: Attention weights for visualization [seq_len, num_frames]
    """
    model.eval()

    # Prepare MIDI features
    midi_tensor = torch.tensor(midi_temporal, dtype=torch.float32).unsqueeze(0).to(device)

    # Get NEWLINE token indices for penalty
    newline_indices = []
    for token in [config.NEWLINE_TOKEN, '<newline>', 'newline']:
        if token in vocab.word2idx:
            newline_indices.append(vocab.word2idx[token])

    # Encode starting words
    tokens = [vocab.sos_idx]
    for word in start_words:
        word_lower = word.lower()
        if word_lower in vocab.word2idx:
            tokens.append(vocab.word2idx[word_lower])
        else:
            tokens.append(vocab.unk_idx)

    generated_tokens = list(start_words)
    all_attention_weights = []
    hidden = None
    consecutive_newlines = 0
    words_since_newline = len(start_words)  # Track words on current line

    with torch.no_grad():
        # Process start sequence to get initial hidden state
        input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
        outputs, hidden, attn_weights = model(input_tensor, midi_tensor, hidden)

        # Store attention weights for start sequence
        all_attention_weights.append(attn_weights[0])  # [seq_len, num_frames]

        # Get logits for last position
        logits = outputs[0, -1, :]

        # Generate tokens one by one
        for _ in range(max_length - len(start_words)):
            # Apply exponential penalty for consecutive NEWLINEs
            penalty = newline_penalty ** consecutive_newlines if consecutive_newlines > 0 else 1.0
            penalty_tokens = newline_indices if penalty > 1.0 else None

            # Sample next token
            next_idx = sample_with_temperature(
                logits.clone(), temperature, top_k,
                repetition_penalty_tokens=penalty_tokens,
                repetition_penalty=penalty
            )

            # Enforce minimum line length - skip NEWLINE if line too short
            if next_idx in newline_indices and words_since_newline < min_line_words:
                # Re-sample excluding NEWLINE tokens
                logits_no_newline = logits.clone()
                for nl_idx in newline_indices:
                    logits_no_newline[nl_idx] = float('-inf')
                next_idx = sample_with_temperature(
                    logits_no_newline, temperature, top_k,
                    repetition_penalty_tokens=penalty_tokens,
                    repetition_penalty=penalty
                )

            # Check for EOS - skip if under min_length
            if next_idx == vocab.eos_idx:
                if len(generated_tokens) >= min_length:
                    break
                else:
                    # Force re-sample excluding EOS
                    logits[vocab.eos_idx] = float('-inf')
                    next_idx = sample_with_temperature(
                        logits.clone(), temperature, top_k,
                        repetition_penalty_tokens=penalty_tokens,
                        repetition_penalty=penalty
                    )

            # Convert to word and add to generated
            word = vocab.idx2word.get(next_idx, config.UNK_TOKEN)
            generated_tokens.append(word)

            # Track consecutive NEWLINEs and words since newline
            if next_idx in newline_indices or word.lower() in ['<newline>', 'newline']:
                consecutive_newlines += 1
                words_since_newline = 0  # Reset word counter on newline
            else:
                consecutive_newlines = 0
                words_since_newline += 1  # Increment word counter

            # Prepare next input
            next_input = torch.tensor([[next_idx]], dtype=torch.long).to(device)

            # Forward pass for single token
            outputs, hidden, attn_weights = model(next_input, midi_tensor, hidden)
            all_attention_weights.append(attn_weights[0])  # [1, num_frames]
            logits = outputs[0, -1, :]

    # Stack attention weights
    attention_weights = torch.cat(all_attention_weights, dim=0)  # [total_len, num_frames]

    # Format output with proper newlines
    text = _format_lyrics(generated_tokens)

    return text, generated_tokens, attention_weights


def _format_lyrics(tokens: List[str]) -> str:
    """
    Format token list into proper lyrics string.
    Replaces NEWLINE tokens with actual newlines.
    """
    lines = []
    current_line = []

    for token in tokens:
        # Check for newline token (handle various formats)
        token_lower = token.lower().strip()
        if token == config.NEWLINE_TOKEN or token_lower == '<newline>' or token_lower == 'newline':
            if current_line:
                lines.append(' '.join(current_line))
                current_line = []
        else:
            current_line.append(token)

    # Add final line
    if current_line:
        lines.append(' '.join(current_line))

    return '\n'.join(lines)


def generate_multiple_samples(
    model,
    vocab: Vocabulary,
    midi_features,
    start_words: List[str],
    num_samples: int = 3,
    is_attention_model: bool = False,
    temperatures: List[float] = None,
    **kwargs
) -> List[Tuple[str, float]]:
    """
    Generate multiple samples with different temperatures.

    Args:
        model: Trained model
        vocab: Vocabulary
        midi_features: MIDI features (global or temporal depending on model)
        start_words: Starting words
        num_samples: Number of samples to generate
        is_attention_model: Whether model uses attention
        temperatures: List of temperatures to use (defaults to varying around config value)

    Returns:
        List of (generated_text, temperature) tuples
    """
    if temperatures is None:
        # Generate with varying temperatures
        base_temp = config.TEMPERATURE
        temperatures = [base_temp * 0.8, base_temp, base_temp * 1.2][:num_samples]

    samples = []

    for temp in temperatures:
        if is_attention_model:
            text, _, _ = generate_lyrics_attention(
                model, vocab, midi_features, start_words,
                temperature=temp, **kwargs
            )
        else:
            text, _ = generate_lyrics_global(
                model, vocab, midi_features, start_words,
                temperature=temp, **kwargs
            )

        samples.append((text, temp))

    return samples


def batch_generate_for_test(
    model,
    vocab: Vocabulary,
    midi_features_dict: dict,
    song_keys: List[str],
    start_words_list: List[List[str]] = None,
    is_attention_model: bool = False,
    device: torch.device = config.DEVICE
) -> dict:
    """
    Generate lyrics for test set evaluation.

    Assignment requirement: 3 different starting words for each melody.

    Args:
        model: Trained model
        vocab: Vocabulary
        midi_features_dict: Dict mapping song_key to MIDI features
        song_keys: List of song keys to generate for
        start_words_list: List of starting word lists (default: config.TEST_START_WORDS)
        is_attention_model: Whether model uses attention
        device: Device to run on

    Returns:
        Dict mapping song_key to list of generated lyrics
    """
    if start_words_list is None:
        # Default: 3 different starting words as per assignment
        start_words_list = [[word] for word in config.TEST_START_WORDS]

    results = {}

    for song_key in song_keys:
        midi_features = midi_features_dict.get(song_key)
        if midi_features is None:
            continue

        song_results = []

        for start_words in start_words_list:
            if is_attention_model:
                text, tokens, attn = generate_lyrics_attention(
                    model, vocab, midi_features, start_words, device=device
                )
                song_results.append({
                    'start_words': start_words,
                    'generated_text': text,
                    'tokens': tokens,
                    'attention_weights': attn.cpu().numpy()
                })
            else:
                text, tokens = generate_lyrics_global(
                    model, vocab, midi_features, start_words, device=device
                )
                song_results.append({
                    'start_words': start_words,
                    'generated_text': text,
                    'tokens': tokens
                })

        results[song_key] = song_results

    return results
