"""
Evaluation module for lyrics generation models.

Computes metrics and generates test outputs as required by assignment:
- Generate lyrics for test set melodies
- Use 3 different starting words per melody
- Compute perplexity and other metrics
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import json

from . import config
from .vocab import Vocabulary
from .model import LyricsLSTMGlobal, LyricsLSTMAttention
from .losses import create_loss_function
from .generate import batch_generate_for_test, generate_lyrics_global, generate_lyrics_attention


def compute_perplexity(
    model,
    data_loader,
    vocab: Vocabulary,
    device: torch.device,
    is_attention_model: bool = False
) -> float:
    """
    Compute perplexity on a dataset.

    Perplexity = exp(average cross-entropy loss)
    Lower is better.
    """
    model.eval()
    criterion = create_loss_function(vocab, device)

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Computing perplexity", leave=False):
            inputs, targets, midi_global, midi_temporal, lengths = batch

            inputs = inputs.to(device)
            targets = targets.to(device)
            midi_global = midi_global.to(device)
            midi_temporal = midi_temporal.to(device)

            if is_attention_model:
                outputs, _, _ = model(inputs, midi_temporal)
            else:
                outputs, _ = model(inputs, midi_global)

            loss = criterion(outputs, targets)
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    perplexity = math.exp(min(avg_loss, 100))  # Cap to prevent overflow

    return perplexity


def compute_diversity_metrics(generated_texts: List[str]) -> Dict[str, float]:
    """
    Compute diversity metrics for generated lyrics.

    Metrics:
    - distinct_1: Ratio of unique unigrams
    - distinct_2: Ratio of unique bigrams
    - avg_length: Average number of tokens
    - vocab_size: Number of unique words used
    """
    all_tokens = []
    all_bigrams = []

    for text in generated_texts:
        tokens = text.lower().split()
        all_tokens.extend(tokens)

        for i in range(len(tokens) - 1):
            all_bigrams.append((tokens[i], tokens[i+1]))

    # Distinct-1: unique unigrams / total unigrams
    distinct_1 = len(set(all_tokens)) / max(len(all_tokens), 1)

    # Distinct-2: unique bigrams / total bigrams
    distinct_2 = len(set(all_bigrams)) / max(len(all_bigrams), 1)

    # Average length
    lengths = [len(text.split()) for text in generated_texts]
    avg_length = np.mean(lengths) if lengths else 0

    # Vocabulary size used
    vocab_size = len(set(all_tokens))

    return {
        'distinct_1': distinct_1,
        'distinct_2': distinct_2,
        'avg_length': avg_length,
        'vocab_size': vocab_size
    }


def compute_structure_metrics(generated_texts: List[str]) -> Dict[str, float]:
    """
    Compute structure-related metrics.

    Metrics:
    - avg_lines: Average number of lines per song
    - avg_line_length: Average words per line
    - line_length_std: Standard deviation of line lengths
    """
    all_line_lengths = []
    num_lines_per_song = []

    for text in generated_texts:
        lines = text.strip().split('\n')
        num_lines_per_song.append(len(lines))

        for line in lines:
            words = line.split()
            if words:
                all_line_lengths.append(len(words))

    return {
        'avg_lines': np.mean(num_lines_per_song) if num_lines_per_song else 0,
        'avg_line_length': np.mean(all_line_lengths) if all_line_lengths else 0,
        'line_length_std': np.std(all_line_lengths) if all_line_lengths else 0
    }


def evaluate_model(
    model,
    test_dataset,
    vocab: Vocabulary,
    device: torch.device,
    is_attention_model: bool = False,
    model_name: str = "model",
    output_dir: Path = None
) -> Dict:
    """
    Full evaluation of a trained model.

    Assignment requirements:
    - Generate for test set melodies
    - Use 3 different starting words
    - Compute metrics

    Args:
        model: Trained model
        test_dataset: Test dataset with MIDI features
        vocab: Vocabulary
        device: Device
        is_attention_model: Whether model uses attention
        model_name: Name for output files
        output_dir: Directory to save results

    Returns:
        Dictionary with all metrics and generated samples
    """
    if output_dir is None:
        output_dir = config.RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    results = {
        'model_name': model_name,
        'metrics': {},
        'generated_samples': []
    }

    # Get MIDI features for all test songs
    if is_attention_model:
        midi_features = test_dataset.midi_temporal
    else:
        midi_features = test_dataset.midi_global

    # Generate for each test song with 3 different starting words
    print(f"\nGenerating lyrics for {len(test_dataset)} test songs...")
    print(f"Starting words: {config.TEST_START_WORDS}")

    all_generated_texts = []

    for idx in tqdm(range(len(test_dataset)), desc="Generating"):
        song_info = test_dataset.get_song_info(idx)
        song_key = f"{song_info['artist']}_{song_info['song']}"

        song_midi = midi_features.get(song_key)
        if song_midi is None:
            continue

        song_generations = []

        for start_word in config.TEST_START_WORDS:
            if is_attention_model:
                text, tokens, attn = generate_lyrics_attention(
                    model, vocab, song_midi, [start_word], device=device
                )
            else:
                text, tokens = generate_lyrics_global(
                    model, vocab, song_midi, [start_word], device=device
                )

            song_generations.append({
                'start_word': start_word,
                'generated_text': text,
                'num_tokens': len(tokens)
            })
            all_generated_texts.append(text)

        results['generated_samples'].append({
            'artist': song_info['artist'],
            'song': song_info['song'],
            'original_lyrics': song_info['lyrics'][:500] + "..." if len(song_info['lyrics']) > 500 else song_info['lyrics'],
            'generations': song_generations
        })

    # Compute metrics
    print("\nComputing metrics...")

    # Diversity metrics
    diversity_metrics = compute_diversity_metrics(all_generated_texts)
    results['metrics']['diversity'] = diversity_metrics

    # Structure metrics
    structure_metrics = compute_structure_metrics(all_generated_texts)
    results['metrics']['structure'] = structure_metrics

    # Summary
    results['metrics']['num_test_songs'] = len(test_dataset)
    results['metrics']['num_generations'] = len(all_generated_texts)

    # Save results
    output_path = output_dir / f"{model_name}_evaluation.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Save generated lyrics as text file for easy reading
    lyrics_path = output_dir / f"{model_name}_generated_lyrics.txt"
    with open(lyrics_path, 'w') as f:
        for sample in results['generated_samples'][:10]:  # First 10 songs
            f.write(f"=" * 60 + "\n")
            f.write(f"Artist: {sample['artist']}\n")
            f.write(f"Song: {sample['song']}\n")
            f.write(f"=" * 60 + "\n\n")

            for gen in sample['generations']:
                f.write(f"--- Starting with: {gen['start_word']} ---\n")
                f.write(gen['generated_text'] + "\n\n")

    print(f"Generated lyrics saved to {lyrics_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Test songs: {results['metrics']['num_test_songs']}")
    print(f"Total generations: {results['metrics']['num_generations']}")
    print(f"\nDiversity Metrics:")
    print(f"  Distinct-1: {diversity_metrics['distinct_1']:.4f}")
    print(f"  Distinct-2: {diversity_metrics['distinct_2']:.4f}")
    print(f"  Vocab size: {diversity_metrics['vocab_size']}")
    print(f"\nStructure Metrics:")
    print(f"  Avg lines/song: {structure_metrics['avg_lines']:.1f}")
    print(f"  Avg words/line: {structure_metrics['avg_line_length']:.1f}")

    return results


def compare_models(
    global_model: LyricsLSTMGlobal,
    attention_model: LyricsLSTMAttention,
    test_dataset,
    vocab: Vocabulary,
    device: torch.device
) -> Dict:
    """
    Compare both model approaches side by side.

    Required by assignment: compare global vs attention approaches.
    """
    print("\n" + "=" * 60)
    print("COMPARING GLOBAL VS ATTENTION MODELS")
    print("=" * 60)

    # Evaluate global model
    print("\n[1/2] Evaluating Global Melody Model...")
    global_results = evaluate_model(
        global_model, test_dataset, vocab, device,
        is_attention_model=False, model_name="global_melody"
    )

    # Evaluate attention model
    print("\n[2/2] Evaluating Attention Melody Model...")
    attention_results = evaluate_model(
        attention_model, test_dataset, vocab, device,
        is_attention_model=True, model_name="attention_melody"
    )

    # Side-by-side comparison
    comparison = {
        'global': global_results['metrics'],
        'attention': attention_results['metrics']
    }

    # Save comparison
    comparison_path = config.RESULTS_DIR / "model_comparison.json"
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<25} {'Global':<15} {'Attention':<15}")
    print("-" * 55)

    for metric_type in ['diversity', 'structure']:
        for metric_name, global_val in global_results['metrics'][metric_type].items():
            attention_val = attention_results['metrics'][metric_type][metric_name]
            print(f"{metric_name:<25} {global_val:<15.4f} {attention_val:<15.4f}")

    return comparison
