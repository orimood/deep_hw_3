"""Lyrics generation with sampling-based text generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path

from . import config
from .data_loader import Vocabulary


def sample_next_word(logits: torch.Tensor,
                     temperature: float = config.TEMPERATURE,
                     top_k: int = config.TOP_K) -> int:
    """
    Sample the next word using temperature-scaled softmax and top-k filtering.

    The sampling is NOT deterministic - words are sampled proportionally
    to their probability, not just picking the argmax.

    Args:
        logits: Logits for vocabulary [vocab_size]
        temperature: Temperature for softmax (lower = more deterministic)
        top_k: Number of top candidates to consider

    Returns:
        Index of sampled word
    """
    # Apply temperature scaling
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

    # Sample from the distribution
    sampled_idx = torch.multinomial(probs, num_samples=1)

    return sampled_idx.item()


def generate_lyrics(model: nn.Module,
                    midi_features: torch.Tensor,
                    midi_temporal: torch.Tensor,
                    vocab: Vocabulary,
                    start_word: str,
                    device: torch.device,
                    max_length: int = 300,  # Safety limit only - model learns structure
                    temperature: float = config.TEMPERATURE,
                    top_k: int = config.TOP_K,
                    is_attention_model: bool = False) -> str:
    """
    Generate lyrics given MIDI features and a starting word.

    The model learns proper line structure (when to insert <NEWLINE>) through
    the structure-aware loss function during training. No hardcoded line limits.

    Args:
        model: Trained lyrics generation model
        midi_features: Global MIDI features [1, midi_dim]
        midi_temporal: Temporal MIDI features [1, num_frames, frame_dim]
        vocab: Vocabulary object
        start_word: Starting word for generation
        device: Device to run on
        max_length: Safety limit for maximum words (model learns actual length)
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        is_attention_model: Whether model uses attention

    Returns:
        Generated lyrics as a string
    """
    model.eval()

    # Special token indices
    sos_idx = vocab.word2idx[config.SOS_TOKEN]
    eos_idx = vocab.word2idx[config.EOS_TOKEN]
    newline_idx = vocab.word2idx[config.NEWLINE_TOKEN]
    unk_idx = vocab.word2idx[config.UNK_TOKEN]

    # Get start word index
    start_word_lower = start_word.lower().strip()
    if start_word_lower in vocab.word2idx:
        start_idx = vocab.word2idx[start_word_lower]
    else:
        print(f"Warning: '{start_word}' not in vocabulary, using <UNK>")
        start_idx = unk_idx

    # Prepare MIDI features
    midi_features = midi_features.to(device)
    midi_temporal = midi_temporal.to(device)

    # Select appropriate MIDI input based on model type
    if is_attention_model:
        midi_input = midi_temporal
    else:
        midi_input = midi_features

    # Initialize generation
    generated_words = [start_word_lower]
    current_word = torch.tensor([[start_idx]], device=device)
    hidden = None

    with torch.no_grad():
        for _ in range(max_length - 1):
            # Forward pass
            if is_attention_model:
                output, hidden, _ = model(current_word, midi_input, hidden)
            else:
                output, hidden = model(current_word, midi_input, hidden)

            # Get logits for next word
            logits = output[0, -1, :]  # [vocab_size]

            # Sample next word
            next_idx = sample_next_word(logits, temperature, top_k)

            # Check for end token
            if next_idx == eos_idx:
                break

            # Get the word
            word = vocab.idx2word.get(next_idx, config.UNK_TOKEN)

            # Handle line breaks - model learns when to insert <NEWLINE> through training
            if word == config.NEWLINE_TOKEN or word.lower() == 'newline':
                generated_words.append('\n')
            else:
                generated_words.append(word)

            # No hardcoded line length limit - model learns structure through loss function

            # Prepare next input
            current_word = torch.tensor([[next_idx]], device=device)

    # Format output
    lyrics = format_lyrics(generated_words)

    return lyrics


def format_lyrics(words: List[str]) -> str:
    """
    Format the generated words into proper lyrics format.

    Args:
        words: List of generated words

    Returns:
        Formatted lyrics string
    """
    lines = []
    current_line = []

    for word in words:
        if word == '\n' or word.lower() == 'newline':
            if current_line:
                lines.append(' '.join(current_line))
                current_line = []
        elif word not in [config.PAD_TOKEN, config.UNK_TOKEN,
                          config.SOS_TOKEN, config.EOS_TOKEN, config.NEWLINE_TOKEN]:
            current_line.append(word)

    # Add any remaining words
    if current_line:
        lines.append(' '.join(current_line))

    return '\n'.join(lines)


def generate_multiple_lyrics(model: nn.Module,
                             midi_features: torch.Tensor,
                             midi_temporal: torch.Tensor,
                             vocab: Vocabulary,
                             start_words: List[str],
                             device: torch.device,
                             is_attention_model: bool = False,
                             **kwargs) -> dict:
    """
    Generate multiple sets of lyrics with different starting words.

    Args:
        model: Trained model
        midi_features: Global MIDI features
        midi_temporal: Temporal MIDI features
        vocab: Vocabulary
        start_words: List of starting words
        device: Device to run on
        is_attention_model: Whether model uses attention
        **kwargs: Additional arguments for generate_lyrics

    Returns:
        Dictionary mapping start words to generated lyrics
    """
    results = {}

    for start_word in start_words:
        print(f"Generating lyrics with start word: '{start_word}'")
        lyrics = generate_lyrics(
            model=model,
            midi_features=midi_features,
            midi_temporal=midi_temporal,
            vocab=vocab,
            start_word=start_word,
            device=device,
            is_attention_model=is_attention_model,
            **kwargs
        )
        results[start_word] = lyrics

    return results


def evaluate_test_set(global_model: nn.Module,
                      attention_model: nn.Module,
                      test_dataset,
                      vocab: Vocabulary,
                      device: torch.device,
                      start_words: List[str] = None,
                      output_dir: str = None) -> dict:
    """
    Evaluate both models on the test set.

    Generates lyrics for each test melody with both approaches
    and multiple starting words.

    Args:
        global_model: Model with global melody conditioning
        attention_model: Model with attention
        test_dataset: Test dataset
        vocab: Vocabulary
        device: Device to run on
        start_words: List of starting words (default: ["love", "the", "i"])
        output_dir: Directory to save results

    Returns:
        Dictionary with all generated lyrics
    """
    if start_words is None:
        start_words = ["love", "the", "i"]

    if output_dir is None:
        output_dir = config.OUTPUTS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for idx in range(len(test_dataset)):
        # Get test sample (now includes both global and temporal MIDI features)
        inputs, targets, midi_features, midi_temporal, length = test_dataset[idx]
        midi_features = midi_features.unsqueeze(0)  # Add batch dimension
        midi_temporal = midi_temporal.unsqueeze(0)  # Add batch dimension

        # Get song info
        song_info = test_dataset.data.iloc[idx]
        song_name = f"{song_info['artist']} - {song_info['song']}"

        print(f"\n{'='*60}")
        print(f"Test Song {idx + 1}: {song_name}")
        print('='*60)

        all_results[song_name] = {}

        # Generate with global model
        print("\n--- Approach 1: Global Melody Conditioning ---")
        global_results = generate_multiple_lyrics(
            model=global_model,
            midi_features=midi_features,
            midi_temporal=midi_temporal,
            vocab=vocab,
            start_words=start_words,
            device=device,
            is_attention_model=False
        )
        all_results[song_name]['global'] = global_results

        for start_word, lyrics in global_results.items():
            print(f"\nStart word: '{start_word}'")
            print("-" * 40)
            print(lyrics)

        # Generate with attention model
        print("\n--- Approach 2: Attention over Melody ---")
        attention_results = generate_multiple_lyrics(
            model=attention_model,
            midi_features=midi_features,
            midi_temporal=midi_temporal,
            vocab=vocab,
            start_words=start_words,
            device=device,
            is_attention_model=True
        )
        all_results[song_name]['attention'] = attention_results

        for start_word, lyrics in attention_results.items():
            print(f"\nStart word: '{start_word}'")
            print("-" * 40)
            print(lyrics)

    # Save results to file
    save_results(all_results, output_dir / "generated_lyrics.txt")

    return all_results


def save_results(results: dict, output_path: Path):
    """Save generated lyrics to a text file."""
    with open(output_path, 'w') as f:
        for song_name, approaches in results.items():
            f.write(f"\n{'='*60}\n")
            f.write(f"Song: {song_name}\n")
            f.write('='*60 + '\n')

            for approach, lyrics_dict in approaches.items():
                f.write(f"\n--- Approach: {approach.upper()} ---\n")

                for start_word, lyrics in lyrics_dict.items():
                    f.write(f"\nStart word: '{start_word}'\n")
                    f.write("-" * 40 + '\n')
                    f.write(lyrics + '\n')

    print(f"\nResults saved to {output_path}")
