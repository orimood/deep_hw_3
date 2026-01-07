"""
Dataset and data loading for lyrics generation.
Patterns from: DebuggerCafe TextDataset, MachineLearningMastery sequence creation.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import gensim.downloader as api
from tqdm import tqdm
import pickle

from . import config
from .vocab import Vocabulary, build_vocab_from_texts
from .midi_features import extract_global_features, extract_temporal_features


class LyricsDataset(Dataset):
    """
    Dataset for lyrics with MIDI features.

    Pattern from DebuggerCafe:
    - input_seq = sequence[:-1]
    - target_seq = sequence[1:]  (shifted by 1)
    """

    def __init__(
        self,
        csv_path: Path,
        midi_dir: Path,
        vocab: Optional[Vocabulary] = None,
        embedding_matrix: Optional[np.ndarray] = None,
        max_seq_length: int = config.MAX_SEQ_LENGTH,
        is_train: bool = True,
        use_cache: bool = True
    ):
        self.midi_dir = Path(midi_dir)
        self.max_seq_length = max_seq_length
        self.is_train = is_train
        self.use_cache = use_cache

        # Ensure cache directory exists
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Load CSV data
        self.data = self._load_csv(csv_path)
        print(f"Loaded {len(self.data)} songs from {csv_path}")

        # Build or load vocabulary
        if vocab is None:
            vocab_cache = config.CACHE_DIR / "vocab.pkl"
            if use_cache and vocab_cache.exists():
                print("Loading vocabulary from cache...")
                self.vocab = Vocabulary.load(vocab_cache)
            else:
                print("Building vocabulary...")
                self.vocab = build_vocab_from_texts(self.data['lyrics'].tolist())
                if use_cache:
                    self.vocab.save(vocab_cache)
        else:
            self.vocab = vocab

        # Build or load embedding matrix
        if embedding_matrix is None:
            embed_cache = config.CACHE_DIR / "embeddings.npz"
            if use_cache and embed_cache.exists():
                print("Loading embeddings from cache...")
                data = np.load(embed_cache)
                self.embedding_matrix = data['embedding_matrix']
            else:
                print("Building embedding matrix...")
                self.embedding_matrix = self._build_embedding_matrix()
                if use_cache:
                    np.savez(embed_cache, embedding_matrix=self.embedding_matrix)
                    print(f"Embeddings saved to {embed_cache}")
        else:
            self.embedding_matrix = embedding_matrix

        # Extract MIDI features (use separate caches for train and test)
        cache_suffix = "train" if is_train else "test"
        midi_cache = config.CACHE_DIR / f"midi_features_{cache_suffix}.pkl"
        if use_cache and midi_cache.exists():
            print("Loading MIDI features from cache...")
            with open(midi_cache, 'rb') as f:
                cache_data = pickle.load(f)
            self.midi_global = cache_data['global']
            self.midi_temporal = cache_data['temporal']
        else:
            print("Extracting MIDI features...")
            self.midi_global, self.midi_temporal = self._extract_all_midi_features()
            if use_cache:
                with open(midi_cache, 'wb') as f:
                    pickle.dump({'global': self.midi_global, 'temporal': self.midi_temporal}, f)
                print(f"MIDI features saved to {midi_cache}")

    def _load_csv(self, csv_path: Path) -> pd.DataFrame:
        """Load and parse the CSV file."""
        df = pd.read_csv(csv_path, header=None)
        # Handle different column counts (train has 7, test has 3)
        if len(df.columns) >= 7:
            df.columns = ['artist', 'song', 'lyrics', 'c4', 'c5', 'c6', 'c7']
        else:
            df.columns = ['artist', 'song', 'lyrics'][:len(df.columns)]
        df = df[['artist', 'song', 'lyrics']].dropna(subset=['lyrics'])

        # Replace ' & ' with NEWLINE token
        df['lyrics'] = df['lyrics'].apply(
            lambda x: str(x).replace(' & ', f' {config.NEWLINE_TOKEN} ')
        )

        return df

    def _build_embedding_matrix(self) -> np.ndarray:
        """
        Build embedding matrix from Word2Vec.
        Pattern from MachineLearningMastery: initialize with random, fill from Word2Vec.
        """
        print(f"Loading Word2Vec model: {config.WORD2VEC_MODEL}")
        word2vec = api.load(config.WORD2VEC_MODEL)

        vocab_size = len(self.vocab)
        embedding_dim = config.EMBEDDING_DIM

        # Initialize with small random values (FloydHub pattern: uniform [-0.1, 0.1])
        embedding_matrix = np.random.uniform(-0.1, 0.1, (vocab_size, embedding_dim)).astype(np.float32)

        # Zero out padding token
        embedding_matrix[self.vocab.pad_idx] = 0

        # Fill from Word2Vec
        found = 0
        for word, idx in self.vocab.word2idx.items():
            if word in word2vec:
                embedding_matrix[idx] = word2vec[word]
                found += 1

        print(f"Found {found}/{vocab_size} words in Word2Vec ({100*found/vocab_size:.1f}%)")
        return embedding_matrix

    def _get_midi_filename(self, artist: str, song: str) -> str:
        """Generate MIDI filename from artist and song."""
        artist_clean = artist.strip().replace(' ', '_').replace("'", "")
        song_clean = song.strip().replace(' ', '_').replace("'", "")
        return f"{artist_clean}_-_{song_clean}.mid"

    def _extract_all_midi_features(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Extract both global and temporal MIDI features for all songs."""
        midi_global = {}
        midi_temporal = {}

        for _, row in tqdm(self.data.iterrows(), total=len(self.data), desc="MIDI features"):
            key = f"{row['artist'].strip()}_{row['song'].strip()}"
            midi_filename = self._get_midi_filename(row['artist'], row['song'])
            midi_path = self.midi_dir / midi_filename

            if midi_path.exists():
                try:
                    midi_global[key] = extract_global_features(str(midi_path))
                    midi_temporal[key] = extract_temporal_features(str(midi_path))
                except Exception as e:
                    print(f"Error processing {midi_filename}: {e}")
                    midi_global[key] = np.zeros(config.MIDI_GLOBAL_DIM, dtype=np.float32)
                    midi_temporal[key] = np.zeros((config.MIDI_TEMPORAL_FRAMES, config.MIDI_FRAME_DIM), dtype=np.float32)
            else:
                # Try alternative filename patterns
                found = False
                artist_clean = row['artist'].strip().lower()
                song_clean = row['song'].strip().lower()
                # Get first word of artist for matching (e.g., "the bangles" -> "bangles")
                artist_words = artist_clean.split()
                artist_key = artist_words[-1] if len(artist_words) > 1 and artist_words[0] == 'the' else artist_words[0]

                for midi_file in self.midi_dir.glob("*.mid"):
                    midi_lower = midi_file.stem.lower()
                    if (song_clean.replace(' ', '_') in midi_lower and
                        artist_key in midi_lower):
                        try:
                            midi_global[key] = extract_global_features(str(midi_file))
                            midi_temporal[key] = extract_temporal_features(str(midi_file))
                            found = True
                            break
                        except Exception:
                            pass

                if not found:
                    midi_global[key] = np.zeros(config.MIDI_GLOBAL_DIM, dtype=np.float32)
                    midi_temporal[key] = np.zeros((config.MIDI_TEMPORAL_FRAMES, config.MIDI_FRAME_DIM), dtype=np.float32)

        return midi_global, midi_temporal

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Get a single sample.

        Returns:
            input_seq: [SOS] + lyrics indices (shifted input)
            target_seq: lyrics indices + [EOS] (shifted target)
            midi_global: Global MIDI features
            midi_temporal: Temporal MIDI features
            length: Sequence length
        """
        row = self.data.iloc[idx]

        # Encode lyrics
        lyrics_indices = self.vocab.encode(row['lyrics'])

        # CRITICAL FIX: Skip samples with empty or very short lyrics
        # These create SOS->EOS patterns that teach the model to stop immediately
        if len(lyrics_indices) < 10:
            # Return a minimal valid sample with padding - will be masked in loss
            # This is better than teaching SOS->EOS
            lyrics_indices = [self.vocab.unk_idx] * 10

        # Truncate if necessary
        if len(lyrics_indices) > self.max_seq_length:
            lyrics_indices = lyrics_indices[:self.max_seq_length]

        # Create input (with SOS) and target (with EOS) - shifted by 1 pattern
        input_seq = [self.vocab.sos_idx] + lyrics_indices
        target_seq = lyrics_indices + [self.vocab.eos_idx]

        # Get MIDI features (strip whitespace for consistent keys)
        key = f"{row['artist'].strip()}_{row['song'].strip()}"
        midi_global = self.midi_global.get(key, np.zeros(config.MIDI_GLOBAL_DIM, dtype=np.float32))
        midi_temporal = self.midi_temporal.get(
            key, np.zeros((config.MIDI_TEMPORAL_FRAMES, config.MIDI_FRAME_DIM), dtype=np.float32)
        )

        return (
            torch.tensor(input_seq, dtype=torch.long),
            torch.tensor(target_seq, dtype=torch.long),
            torch.tensor(midi_global, dtype=torch.float32),
            torch.tensor(midi_temporal, dtype=torch.float32),
            len(input_seq)
        )

    def get_song_info(self, idx: int) -> Dict:
        """Get metadata for a song."""
        row = self.data.iloc[idx]
        return {
            'artist': row['artist'].strip(),
            'song': row['song'].strip(),
            'lyrics': row['lyrics']
        }


def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, ...]:
    """
    Custom collate function for variable length sequences.
    Pads sequences to the same length within the batch.
    """
    inputs, targets, midi_global, midi_temporal, lengths = zip(*batch)

    # Pad sequences (DebuggerCafe pattern)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)

    # Stack tensors
    midi_global = torch.stack(midi_global)
    midi_temporal = torch.stack(midi_temporal)
    lengths = torch.tensor(lengths, dtype=torch.long)

    return inputs_padded, targets_padded, midi_global, midi_temporal, lengths


def get_dataloaders(
    batch_size: int = config.BATCH_SIZE,
    val_split: float = config.VALIDATION_SPLIT,
    use_cache: bool = True
) -> Tuple[DataLoader, DataLoader, LyricsDataset]:
    """
    Create train and validation dataloaders.

    Returns:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        train_dataset: Full training dataset (for access to vocab, embeddings)
    """
    # Create training dataset
    train_dataset = LyricsDataset(
        csv_path=config.TRAIN_CSV,
        midi_dir=config.MIDI_DIR,
        is_train=True,
        use_cache=use_cache
    )

    # Split into train and validation
    total_size = len(train_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.SEED)
    )

    print(f"Train size: {train_size}, Validation size: {val_size}")

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    return train_loader, val_loader, train_dataset


def get_test_dataset(train_dataset: LyricsDataset) -> LyricsDataset:
    """Create test dataset using vocabulary from training."""
    return LyricsDataset(
        csv_path=config.TEST_CSV,
        midi_dir=config.MIDI_DIR,
        vocab=train_dataset.vocab,
        embedding_matrix=train_dataset.embedding_matrix,
        is_train=False,
        use_cache=True
    )


def get_curriculum_dataloaders(
    batch_size: int = config.BATCH_SIZE,
    val_split: float = config.VALIDATION_SPLIT,
    min_words: int = 100,
    use_cache: bool = True
) -> Tuple[DataLoader, DataLoader, LyricsDataset]:
    """
    Create dataloaders filtered for longer sequences (curriculum learning).

    Pre-training on longer songs first teaches the model to produce
    longer outputs before seeing shorter examples.

    Args:
        batch_size: Batch size
        val_split: Validation split ratio
        min_words: Minimum words per song to include
        use_cache: Whether to use cached data

    Returns:
        train_loader: Training DataLoader (filtered for long songs)
        val_loader: Validation DataLoader (filtered for long songs)
        full_dataset: Full training dataset (for vocab, embeddings)
    """
    # Create full training dataset
    full_dataset = LyricsDataset(
        csv_path=config.TRAIN_CSV,
        midi_dir=config.MIDI_DIR,
        is_train=True,
        use_cache=use_cache
    )

    # Filter indices for songs with >= min_words
    long_indices = []
    for idx in range(len(full_dataset)):
        lyrics = full_dataset.data.iloc[idx]['lyrics']
        word_count = len(str(lyrics).split())
        if word_count >= min_words:
            long_indices.append(idx)

    print(f"Curriculum learning: {len(long_indices)}/{len(full_dataset)} songs have >= {min_words} words")

    # Create subset with only long songs
    long_subset = torch.utils.data.Subset(full_dataset, long_indices)

    # Split into train and validation
    total_size = len(long_subset)
    val_size = max(1, int(total_size * val_split))
    train_size = total_size - val_size

    train_subset, val_subset = torch.utils.data.random_split(
        long_subset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.SEED)
    )

    print(f"Curriculum train size: {train_size}, Curriculum val size: {val_size}")

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    return train_loader, val_loader, full_dataset
