#!/usr/bin/env python3
"""
Interactive lyrics generation script.

Usage:
    python run_generate.py --approach global --midi path/to/song.mid --start-words "love is"
    python run_generate.py --approach attention --song-idx 0 --start-words "the night"
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config
from src.dataset import get_dataloaders, get_test_dataset
from src.model import create_model
from src.train import load_checkpoint
from src.generate import generate_lyrics_global, generate_lyrics_attention
from src.midi_features import extract_global_features, extract_temporal_features


def main():
    parser = argparse.ArgumentParser(description='Generate lyrics from trained model')
    parser.add_argument(
        '--approach',
        type=str,
        choices=['global', 'attention'],
        default='global',
        help='Which model approach to use'
    )
    parser.add_argument(
        '--midi',
        type=str,
        default=None,
        help='Path to MIDI file for conditioning'
    )
    parser.add_argument(
        '--song-idx',
        type=int,
        default=0,
        help='Index of test song to use for MIDI features'
    )
    parser.add_argument(
        '--start-words',
        type=str,
        default='love',
        help='Starting words for generation (space-separated)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=config.TEMPERATURE,
        help='Sampling temperature (higher = more random)'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=config.MAX_GENERATION_LENGTH,
        help='Maximum number of tokens to generate'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=1,
        help='Number of samples to generate'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint'
    )
    args = parser.parse_args()

    print("=" * 60)
    print("LYRICS GENERATION")
    print("=" * 60)

    # Load data to get vocab
    print("\nLoading vocabulary...")
    _, _, train_dataset = get_dataloaders(use_cache=True)
    vocab = train_dataset.vocab
    embedding_matrix = train_dataset.embedding_matrix

    # Create and load model
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
        print("Using untrained model (results will be random)")

    # Get MIDI features
    if args.midi:
        print(f"\nExtracting features from: {args.midi}")
        if args.approach == 'global':
            midi_features = extract_global_features(args.midi)
        else:
            midi_features = extract_temporal_features(args.midi)
    else:
        print(f"\nUsing test song {args.song_idx} for MIDI features...")
        test_dataset = get_test_dataset(train_dataset)
        song_info = test_dataset.get_song_info(args.song_idx)
        song_key = f"{song_info['artist']}_{song_info['song']}"

        print(f"Song: {song_info['artist']} - {song_info['song']}")

        if args.approach == 'global':
            midi_features = test_dataset.midi_global.get(
                song_key,
                np.zeros(config.MIDI_GLOBAL_DIM, dtype=np.float32)
            )
        else:
            midi_features = test_dataset.midi_temporal.get(
                song_key,
                np.zeros((config.MIDI_TEMPORAL_FRAMES, config.MIDI_FRAME_DIM), dtype=np.float32)
            )

    # Parse starting words
    start_words = args.start_words.split()
    print(f"\nStarting words: {start_words}")
    print(f"Temperature: {args.temperature}")
    print(f"Max length: {args.max_length}")

    # Generate
    print("\n" + "-" * 60)
    print("GENERATED LYRICS")
    print("-" * 60)

    for i in range(args.num_samples):
        if args.num_samples > 1:
            print(f"\n[Sample {i+1}]")

        if args.approach == 'global':
            text, tokens = generate_lyrics_global(
                model=model,
                vocab=vocab,
                midi_features=midi_features,
                start_words=start_words,
                max_length=args.max_length,
                temperature=args.temperature,
                device=config.DEVICE
            )
        else:
            text, tokens, attn = generate_lyrics_attention(
                model=model,
                vocab=vocab,
                midi_temporal=midi_features,
                start_words=start_words,
                max_length=args.max_length,
                temperature=args.temperature,
                device=config.DEVICE
            )

        print(f"\n{text}")
        print(f"\n({len(tokens)} tokens generated)")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
