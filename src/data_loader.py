"""Data loading and preprocessing for lyrics and MIDI files."""

import re
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from pathlib import Path
import gensim.downloader as api
from typing import Dict, List, Tuple, Optional
import pretty_midi
from tqdm import tqdm

from . import config
from .midi_features import extract_midi_features


class Vocabulary:
    """Vocabulary class for mapping words to indices."""

    def __init__(self):
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_count: Counter = Counter()

        # Add special tokens
        for token in [config.PAD_TOKEN, config.UNK_TOKEN,
                      config.SOS_TOKEN, config.EOS_TOKEN, config.NEWLINE_TOKEN]:
            self._add_word(token)

    def _add_word(self, word: str) -> int:
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        return self.word2idx[word]

    def add_sentence(self, sentence: str):
        """Add words from a sentence to the vocabulary."""
        words = self.tokenize(sentence)
        for word in words:
            self._add_word(word)
            self.word_count[word] += 1

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize text into words."""
        # Convert to lowercase and split on whitespace
        text = text.lower().strip()
        # Simple tokenization - split on spaces and remove empty strings
        words = text.split()
        # Clean words - remove punctuation attached to words
        cleaned = []
        for word in words:
            # Keep basic punctuation as separate tokens
            word = re.sub(r"[^\w\s'&]", "", word)
            if word:
                cleaned.append(word)
        return cleaned

    def encode(self, sentence: str) -> List[int]:
        """Encode a sentence to indices."""
        words = self.tokenize(sentence)
        return [self.word2idx.get(w, self.word2idx[config.UNK_TOKEN]) for w in words]

    def decode(self, indices: List[int]) -> str:
        """Decode indices to words."""
        words = [self.idx2word.get(idx, config.UNK_TOKEN) for idx in indices]
        return " ".join(words)

    def __len__(self):
        return len(self.word2idx)


class LyricsDataset(Dataset):
    """Dataset for lyrics with MIDI features."""

    def __init__(self,
                 csv_path: str,
                 midi_dir: str,
                 vocab: Optional[Vocabulary] = None,
                 word2vec_model = None,
                 is_train: bool = True,
                 max_seq_length: int = 200):

        self.midi_dir = Path(midi_dir)
        self.is_train = is_train
        self.max_seq_length = max_seq_length

        # Load data
        self.data = self._load_csv(csv_path)

        # Build or use vocabulary
        if vocab is None:
            self.vocab = Vocabulary()
            for _, row in self.data.iterrows():
                self.vocab.add_sentence(row['lyrics'])
        else:
            self.vocab = vocab

        # Load Word2Vec model
        self.word2vec_model = word2vec_model
        if self.word2vec_model is None:
            print("Loading Word2Vec model...")
            self.word2vec_model = api.load(config.WORD2VEC_MODEL)

        # Build word embedding matrix
        self.embedding_matrix = self._build_embedding_matrix()

        # Pre-extract MIDI features for all songs
        print("Extracting MIDI features...")
        self.midi_features = self._extract_all_midi_features()

    def _load_csv(self, csv_path: str) -> pd.DataFrame:
        """Load and parse the CSV file."""
        # Read CSV without header
        df = pd.read_csv(csv_path, header=None, names=['artist', 'song', 'lyrics', 'c4', 'c5', 'c6', 'c7'])

        # Keep only relevant columns
        df = df[['artist', 'song', 'lyrics']].dropna(subset=['lyrics'])

        # Clean lyrics - replace & with newline token
        df['lyrics'] = df['lyrics'].apply(lambda x: str(x).replace(' & ', f' {config.NEWLINE_TOKEN} '))

        return df

    def _get_midi_filename(self, artist: str, song: str) -> str:
        """Generate MIDI filename from artist and song."""
        # Clean and format the filename
        artist_clean = artist.strip().replace(' ', '_').replace("'", "")
        song_clean = song.strip().replace(' ', '_').replace("'", "")
        return f"{artist_clean}_-_{song_clean}.mid"

    def _build_embedding_matrix(self) -> np.ndarray:
        """Build embedding matrix from Word2Vec."""
        vocab_size = len(self.vocab)
        embedding_dim = config.WORD2VEC_DIM

        # Initialize with random vectors
        embedding_matrix = np.random.randn(vocab_size, embedding_dim) * 0.01

        # Zero out padding token
        embedding_matrix[self.vocab.word2idx[config.PAD_TOKEN]] = 0

        found_count = 0
        for word, idx in self.vocab.word2idx.items():
            if word in self.word2vec_model:
                embedding_matrix[idx] = self.word2vec_model[word]
                found_count += 1

        print(f"Found {found_count}/{vocab_size} words in Word2Vec model")
        return embedding_matrix

    def _extract_all_midi_features(self) -> Dict[str, np.ndarray]:
        """Extract MIDI features for all songs."""
        midi_features = {}

        for _, row in tqdm(self.data.iterrows(), total=len(self.data)):
            midi_filename = self._get_midi_filename(row['artist'], row['song'])
            midi_path = self.midi_dir / midi_filename

            key = f"{row['artist']}_{row['song']}"

            if midi_path.exists():
                try:
                    features = extract_midi_features(str(midi_path))
                    midi_features[key] = features
                except Exception as e:
                    print(f"Error processing {midi_filename}: {e}")
                    midi_features[key] = np.zeros(config.MIDI_FEATURE_DIM)
            else:
                # Try alternative filename formats
                found = False
                for midi_file in self.midi_dir.glob("*.mid"):
                    if (row['song'].lower().replace(' ', '_') in midi_file.stem.lower() and
                        row['artist'].lower().replace(' ', '_') in midi_file.stem.lower()):
                        try:
                            features = extract_midi_features(str(midi_file))
                            midi_features[key] = features
                            found = True
                            break
                        except Exception as e:
                            pass

                if not found:
                    midi_features[key] = np.zeros(config.MIDI_FEATURE_DIM)

        return midi_features

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.data.iloc[idx]

        # Encode lyrics
        lyrics_indices = self.vocab.encode(row['lyrics'])

        # Truncate if necessary
        if len(lyrics_indices) > self.max_seq_length:
            lyrics_indices = lyrics_indices[:self.max_seq_length]

        # Create input (with SOS) and target (with EOS)
        sos_idx = self.vocab.word2idx[config.SOS_TOKEN]
        eos_idx = self.vocab.word2idx[config.EOS_TOKEN]

        input_seq = [sos_idx] + lyrics_indices
        target_seq = lyrics_indices + [eos_idx]

        # Get MIDI features
        key = f"{row['artist']}_{row['song']}"
        midi_feat = self.midi_features.get(key, np.zeros(config.MIDI_FEATURE_DIM))

        return (
            torch.tensor(input_seq, dtype=torch.long),
            torch.tensor(target_seq, dtype=torch.long),
            torch.tensor(midi_feat, dtype=torch.float32),
            torch.tensor(len(input_seq), dtype=torch.long)
        )


def collate_fn(batch):
    """Custom collate function for variable length sequences."""
    inputs, targets, midi_feats, lengths = zip(*batch)

    # Pad sequences
    pad_idx = 0  # PAD token index
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=pad_idx)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=pad_idx)

    midi_feats = torch.stack(midi_feats)
    lengths = torch.stack(lengths)

    return inputs_padded, targets_padded, midi_feats, lengths


def get_dataloaders(batch_size: int = config.BATCH_SIZE,
                    val_split: float = config.VALIDATION_SPLIT):
    """Create train and validation dataloaders."""

    # Load Word2Vec model once
    print("Loading Word2Vec model...")
    word2vec_model = api.load(config.WORD2VEC_MODEL)

    # Create training dataset
    train_dataset = LyricsDataset(
        csv_path=str(config.TRAIN_CSV),
        midi_dir=str(config.MIDI_DIR),
        word2vec_model=word2vec_model,
        is_train=True
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


def get_test_dataset(train_dataset: LyricsDataset):
    """Create test dataset using vocabulary from training."""
    test_dataset = LyricsDataset(
        csv_path=str(config.TEST_CSV),
        midi_dir=str(config.MIDI_DIR),
        vocab=train_dataset.vocab,
        word2vec_model=train_dataset.word2vec_model,
        is_train=False
    )
    return test_dataset
