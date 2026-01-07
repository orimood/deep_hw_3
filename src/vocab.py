"""
Vocabulary handling for word-level text generation.
Source patterns: DebuggerCafe, MachineLearningMastery
"""

import re
import pickle
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional

from . import config


class Vocabulary:
    """
    Vocabulary class for mapping words to indices and vice versa.

    Features:
    - Special tokens (PAD, UNK, SOS, EOS, NEWLINE)
    - Word frequency tracking
    - Optional frequency filtering (MachineLearningMastery pattern)
    - Serialization support
    """

    def __init__(self, min_freq: int = config.MIN_WORD_FREQUENCY):
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_count: Counter = Counter()
        self.min_freq = min_freq
        self._finalized = False

        # Add special tokens first (they always get lowest indices)
        for token in config.SPECIAL_TOKENS:
            self._add_word(token)

    def _add_word(self, word: str) -> int:
        """Add a word to vocabulary, return its index."""
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        return self.word2idx[word]

    def add_sentence(self, sentence: str) -> None:
        """Add words from a sentence to vocabulary."""
        words = self.tokenize(sentence)
        for word in words:
            self.word_count[word] += 1

    def finalize(self) -> None:
        """
        Finalize vocabulary by filtering rare words.
        Call this after adding all sentences.
        Pattern from MachineLearningMastery: sort by frequency
        """
        if self._finalized:
            return

        # Get words sorted by frequency (most common first)
        sorted_words = [w for w, c in self.word_count.most_common()
                       if c >= self.min_freq and w not in config.SPECIAL_TOKENS]

        # Add to vocabulary (special tokens already added)
        for word in sorted_words:
            self._add_word(word)

        self._finalized = True
        print(f"Vocabulary finalized: {len(self)} words")
        print(f"  - Special tokens: {len(config.SPECIAL_TOKENS)}")
        print(f"  - Regular words: {len(self) - len(config.SPECIAL_TOKENS)}")

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Tokenize text into words.
        Pattern from DebuggerCafe: simple split with cleaning.

        IMPORTANT: Preserves special tokens like <NEWLINE> before cleaning.
        """
        text = text.strip()

        # CRITICAL FIX: Preserve NEWLINE tokens before lowercase/cleaning
        # Replace <NEWLINE> with a placeholder that survives the regex
        newline_placeholder = "___NEWLINE___"
        text = text.replace(config.NEWLINE_TOKEN, newline_placeholder)

        text = text.lower()
        # Remove most punctuation but keep apostrophes
        text = re.sub(r"[^\w\s']", " ", text)

        # Split and filter empty strings
        words = [w for w in text.split() if w]

        # CRITICAL FIX: Restore NEWLINE tokens (placeholder -> original token)
        words = [config.NEWLINE_TOKEN if w == newline_placeholder.lower() else w for w in words]

        return words

    def encode(self, sentence: str) -> List[int]:
        """Encode a sentence to list of indices."""
        words = self.tokenize(sentence)
        unk_idx = self.word2idx[config.UNK_TOKEN]
        return [self.word2idx.get(w, unk_idx) for w in words]

    def decode(self, indices: List[int], skip_special: bool = True) -> str:
        """Decode indices back to words.

        FIXED: Properly handles newlines without adding extra spaces.
        """
        skip_tokens = {config.PAD_TOKEN, config.SOS_TOKEN, config.EOS_TOKEN} if skip_special else set()

        # Build lines instead of a flat word list
        lines = []
        current_line = []

        for idx in indices:
            word = self.idx2word.get(idx, config.UNK_TOKEN)
            if word in skip_tokens:
                continue

            if word == config.NEWLINE_TOKEN:
                # Finish current line and start new one
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = []
                else:
                    # Empty line (consecutive newlines)
                    lines.append('')
            else:
                current_line.append(word)

        # Don't forget the last line
        if current_line:
            lines.append(' '.join(current_line))

        return '\n'.join(lines)

    def __len__(self) -> int:
        return len(self.word2idx)

    def __contains__(self, word: str) -> bool:
        return word in self.word2idx

    # Token index properties for convenience
    @property
    def pad_idx(self) -> int:
        return self.word2idx[config.PAD_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.word2idx[config.UNK_TOKEN]

    @property
    def sos_idx(self) -> int:
        return self.word2idx[config.SOS_TOKEN]

    @property
    def eos_idx(self) -> int:
        return self.word2idx[config.EOS_TOKEN]

    @property
    def newline_idx(self) -> int:
        return self.word2idx[config.NEWLINE_TOKEN]

    def save(self, path: Path) -> None:
        """Save vocabulary to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'word_count': dict(self.word_count),
                'min_freq': self.min_freq,
                'finalized': self._finalized
            }, f)
        print(f"Vocabulary saved to {path}")

    @classmethod
    def load(cls, path: Path) -> 'Vocabulary':
        """Load vocabulary from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        vocab = cls.__new__(cls)
        vocab.word2idx = data['word2idx']
        vocab.idx2word = data['idx2word']
        vocab.word_count = Counter(data['word_count'])
        vocab.min_freq = data['min_freq']
        vocab._finalized = data['finalized']

        print(f"Vocabulary loaded from {path}: {len(vocab)} words")
        return vocab


def build_vocab_from_texts(texts: List[str], min_freq: int = config.MIN_WORD_FREQUENCY) -> Vocabulary:
    """
    Build vocabulary from list of text strings.

    Args:
        texts: List of text documents
        min_freq: Minimum word frequency threshold

    Returns:
        Finalized Vocabulary object
    """
    vocab = Vocabulary(min_freq=min_freq)

    for text in texts:
        vocab.add_sentence(text)

    vocab.finalize()
    return vocab
