#!/usr/bin/env python3
"""
Evaluation script for GRU-based lyrics generation models.

Generates lyrics using trained GRU models and saves results.
"""

import torch
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config
from src.data_loader import LyricsDataset, Vocabulary
from src.generate import generate_lyrics, format_lyrics
from gru_models import create_gru_model

import torch.nn.functional as F


def load_gru_model(approach: str, vocab_size: int, embedding_matrix, device: torch.device):
    """Load a trained GRU model."""
    models_dir = Path(__file__).parent / "models"
    model_path = models_dir / f"lyrics_gru_{approach}_best.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = create_gru_model(
        approach=approach,
        vocab_size=vocab_size,
        embedding_matrix=embedding_matrix,
        device=device
    )

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded checkpoint from {model_path}")
    print(f"  Epoch: {checkpoint['epoch'] + 1}, Val Loss: {checkpoint['val_loss']:.4f}")

    return model


def sample_next_word(logits: torch.Tensor,
                     temperature: float = config.TEMPERATURE,
                     top_k: int = config.TOP_K) -> int:
    """Sample next word using temperature and top-k."""
    logits = logits / temperature

    if top_k > 0:
        top_k_values, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
        logits_filtered = torch.full_like(logits, float('-inf'))
        logits_filtered.scatter_(0, top_k_indices, top_k_values)
        logits = logits_filtered

    probs = F.softmax(logits, dim=-1)
    sampled_idx = torch.multinomial(probs, num_samples=1)

    return sampled_idx.item()


def generate_gru_lyrics(model, midi_features, midi_temporal, vocab, start_word,
                        device, max_length=300, temperature=config.TEMPERATURE,
                        top_k=config.TOP_K, is_attention_model=False):
    """Generate lyrics using GRU model."""
    model.eval()

    sos_idx = vocab.word2idx[config.SOS_TOKEN]
    eos_idx = vocab.word2idx[config.EOS_TOKEN]
    newline_idx = vocab.word2idx[config.NEWLINE_TOKEN]
    unk_idx = vocab.word2idx[config.UNK_TOKEN]

    start_word_lower = start_word.lower().strip()
    if start_word_lower in vocab.word2idx:
        start_idx = vocab.word2idx[start_word_lower]
    else:
        print(f"Warning: '{start_word}' not in vocabulary, using <UNK>")
        start_idx = unk_idx

    midi_features = midi_features.to(device)
    midi_temporal = midi_temporal.to(device)

    if is_attention_model:
        midi_input = midi_temporal
    else:
        midi_input = midi_features

    generated_words = [start_word_lower]
    current_word = torch.tensor([[start_idx]], device=device)
    hidden = None

    with torch.no_grad():
        for _ in range(max_length - 1):
            if is_attention_model:
                output, hidden, _ = model(current_word, midi_input, hidden)
            else:
                output, hidden = model(current_word, midi_input, hidden)

            logits = output[0, -1, :]
            next_idx = sample_next_word(logits, temperature, top_k)

            if next_idx == eos_idx:
                break

            word = vocab.idx2word.get(next_idx, config.UNK_TOKEN)

            if word == config.NEWLINE_TOKEN or word.lower() == 'newline':
                generated_words.append('\n')
            else:
                generated_words.append(word)

            current_word = torch.tensor([[next_idx]], device=device)

    lyrics = format_lyrics(generated_words)
    return lyrics


def main():
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    start_words = ["love", "the", "i"]
    print(f"Starting words: {start_words}")
    print(f"Using device: {device}")

    # Load data
    print("\n" + "="*60)
    print("Loading data...")
    print("="*60)

    train_dataset = LyricsDataset(csv_path=config.TRAIN_CSV)
    test_dataset = LyricsDataset(
        csv_path=config.TEST_CSV,
        vocab=train_dataset.vocab,
        embedding_matrix=train_dataset.embedding_matrix
    )

    vocab = train_dataset.vocab
    embedding_matrix = train_dataset.embedding_matrix
    vocab_size = len(vocab)

    print(f"Vocabulary size: {vocab_size}")
    print(f"Test songs: {len(test_dataset)}")

    # Load models
    models_dir = Path(__file__).parent / "models"

    models = {}
    for approach in ["global", "attention"]:
        model_path = models_dir / f"lyrics_gru_{approach}_best.pt"
        if model_path.exists():
            print(f"\n" + "="*60)
            print(f"Loading GRU {approach.upper()} model...")
            print("="*60)
            models[approach] = load_gru_model(approach, vocab_size, embedding_matrix, device)
        else:
            print(f"Warning: GRU {approach} model not found, skipping...")

    if not models:
        print("No models found! Please train models first.")
        return

    # Generate lyrics
    print("\n" + "="*60)
    print("Generating lyrics for test set...")
    print("="*60)

    all_results = {}

    for idx in range(len(test_dataset)):
        inputs, targets, midi_features, midi_temporal, length = test_dataset[idx]
        midi_features = midi_features.unsqueeze(0)
        midi_temporal = midi_temporal.unsqueeze(0)

        song_info = test_dataset.data.iloc[idx]
        song_name = f"{song_info['artist']} - {song_info['song']}"

        print(f"\n{'='*60}")
        print(f"Test Song {idx + 1}: {song_name}")
        print('='*60)

        all_results[song_name] = {}

        for approach, model in models.items():
            is_attention = (approach == "attention")
            print(f"\n--- GRU {approach.upper()} ---")

            approach_results = {}
            for start_word in start_words:
                print(f"Generating lyrics with start word: '{start_word}'")
                lyrics = generate_gru_lyrics(
                    model=model,
                    midi_features=midi_features,
                    midi_temporal=midi_temporal,
                    vocab=vocab,
                    start_word=start_word,
                    device=device,
                    is_attention_model=is_attention
                )
                approach_results[start_word] = lyrics

            all_results[song_name][f'gru_{approach}'] = approach_results

            for start_word, lyrics in approach_results.items():
                print(f"\nStart word: '{start_word}'")
                print("-" * 40)
                print(lyrics)

    # Save results
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "gru_generated_lyrics.txt"

    with open(output_path, 'w') as f:
        for song_name, approaches in all_results.items():
            f.write(f"\n{'='*60}\n")
            f.write(f"Song: {song_name}\n")
            f.write('='*60 + '\n')

            for approach, lyrics_dict in approaches.items():
                f.write(f"\n--- {approach.upper()} ---\n")

                for start_word, lyrics in lyrics_dict.items():
                    f.write(f"\nStart word: '{start_word}'\n")
                    f.write("-" * 40 + '\n')
                    f.write(lyrics + '\n')

    print(f"\nResults saved to {output_path}")

    print("\n" + "="*60)
    print("GRU Evaluation complete!")
    print("="*60)


if __name__ == "__main__":
    main()
