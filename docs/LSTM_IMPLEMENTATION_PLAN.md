# LSTM Word-Level Text Generation Implementation Plan

## Overview

This plan details how to implement an LSTM-based word-level text generation system for song lyrics, adapted for PyTorch and integrated with the project's melody conditioning requirements.

---

## Assignment Requirements Compliance

### Mandatory Requirements (from Assignment 3)

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| **PyTorch** | ✅ | All code in PyTorch |
| **LSTM or GRU** | ✅ | Using LSTM (`nn.LSTM`) |
| **Word2Vec (300-dim)** | ✅ | `gensim` Word2Vec, 300 dimensions |
| **Predict next word** | ✅ | Standard LM: input[t] → predict target[t+1] |
| **MIDI integration** | ✅ | `pretty_midi` for feature extraction |
| **TWO approaches for melody** | ✅ | Approach 1: Global, Approach 2: Attention |
| **Sampling-based generation** | ✅ | `torch.multinomial()` with temperature |
| **Length guidelines via loss** | ✅ | `StructureAwareLoss` in `src/losses.py` |
| **TensorBoard logging** | ✅ | Training/validation loss curves |

### Test Phase Requirements

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Generate for each test melody | ✅ | `evaluate_test_set()` in `src/generate.py` |
| Both architectural variants | ✅ | Global model + Attention model |
| **3 different starting words** | ✅ | Default: `["love", "the", "i"]` |
| Same words for all melodies | ✅ | Configured in `run_evaluation.py` |
| Analyze first word effect | 📝 | To include in report |
| Analyze melody effect | 📝 | To include in report |

### Report Requirements

| Section | Content Needed |
|---------|----------------|
| Architecture | Describe LSTM + Word2Vec + MIDI features |
| Melody Integration | Explain Global (concat) vs Attention (dynamic) |
| TensorBoard Graphs | Training loss, validation loss for both models |
| Generated Lyrics | All outputs: 4 melodies × 3 words × 2 approaches |
| Analysis | Effect of starting word, effect of melody |

---

## Sources Referenced

| Source | Key Insights |
|--------|--------------|
| [enriqueav/lstm_lyrics](https://github.com/enriqueav/lstm_lyrics/blob/master/lstm_train_embedding.py) | Bidirectional LSTM, 1024-dim embedding, sequence prep with STEP=1 |
| [TensorFlow Text Generation Tutorial](https://www.tensorflow.org/text/tutorials/text_generation) | GRU architecture, embedding_dim=256, rnn_units=1024, temperature sampling |
| [DebuggerCafe Word-Level LSTM](https://debuggercafe.com/word-level-text-generation-using-lstm/) | Simple LSTM: embed_dim=16, hidden=32, seq_len=64 |
| [MachineLearningMastery Word-Level LM](https://machinelearningmastery.com/how-to-develop-a-word-level-neural-language-model-in-keras/) | 2 LSTM layers (100 units each), 50-dim embedding, seq_len=50 |
| [FloydHub word-language-model](https://github.com/floydhub/word-language-model) | PyTorch reference: embed=128, hidden=128, nlayers=3, dropout=0.2 |

### Consolidated Best Practices from All Sources

| Aspect | Range Across Sources | Chosen for Project | Rationale |
|--------|---------------------|-------------------|-----------|
| Embedding Dim | 16-1024 | 300 (Word2Vec) | Pre-trained captures semantics |
| Hidden Units | 32-1024 | 512 | Balance capacity vs. training speed |
| LSTM Layers | 1-3 | 2 | Sufficient depth without overfitting |
| Dropout | 0.2-0.65 | 0.3 | Regularization for small dataset |
| Sequence Length | 10-100 | 200 (truncated) | Full song context |
| Batch Size | 20-64 | 32 | Standard choice |
| Temperature | 0.7-1.0 | 0.8 | Creative but coherent |
| Top-k | 50 | 50 | Filter unlikely words |
| Gradient Clip | 0.25-5.0 | 5.0 | Prevent exploding gradients |

---

## Source 1: FloydHub word-language-model (PyTorch Reference)

### Complete Model Architecture

```python
import torch.nn as nn
from torch.autograd import Variable

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)  # Token -> Embeddings

        # Supports LSTM, GRU, RNN_TANH, RNN_RELU
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)

        self.decoder = nn.Linear(nhid, ntoken)

        # Weight tying: share embedding and output weights
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using tied flag, nhid must equal emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        """Initialize weights uniformly in range [-0.1, 0.1]"""
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))  # Dropout on embeddings
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)  # Dropout on RNN output
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        """Initialize hidden state on same device as model weights"""
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
```

### FloydHub Training Configuration

```python
# Default hyperparameters from main.py
args = {
    'emsize': 200,      # Embedding dimension
    'nhid': 200,        # Hidden units per layer
    'nlayers': 2,       # Number of LSTM layers
    'lr': 20,           # Initial learning rate (high for SGD)
    'clip': 0.25,       # Gradient clipping threshold
    'epochs': 40,       # Max epochs
    'batch_size': 20,   # Batch size
    'bptt': 35,         # Sequence length (backprop through time)
    'dropout': 0.2,     # Dropout rate
    'tied': False,      # Weight tying flag
}

# Learning rate decay: divide by 4 when val loss doesn't improve
# Gradient clipping: torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
```

### Key Patterns to Adopt

1. **Dropout on both embeddings AND RNN output** (not just between layers)
2. **Weight initialization** with uniform distribution [-0.1, 0.1]
3. **Weight tying** option (embedding weights = output weights)
4. **Learning rate decay** when validation loss plateaus

---

## Source 2: DebuggerCafe (PyTorch Tutorial)

### Complete Model Class

```python
class TextGenerationLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers):
        super(TextGenerationLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True  # Important: (batch, seq, features)
        )
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(x.shape[0])
        x = self.embedding(x)
        out, (h_n, c_n) = self.lstm(x, hidden)
        out = out.contiguous().view(-1, self.hidden_size)  # Flatten for FC
        out = self.fc(out)
        return out, (h_n, c_n)

    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return h0, c0
```

### Data Preparation Pattern

```python
# Vocabulary building
words = text.split()
word_counts = Counter(words)
vocab = list(word_counts.keys())
word_to_int = {word: i for i, word in enumerate(vocab)}
int_to_word = {i: word for word, i in word_to_int.items()}

# Sequence creation: input[:-1] -> target[1:] (shifted by 1)
SEQUENCE_LENGTH = 64
samples = [words[i:i+SEQUENCE_LENGTH+1] for i in range(len(words)-SEQUENCE_LENGTH)]
```

### Dataset Class Pattern

```python
class TextDataset(Dataset):
    def __init__(self, samples, word_to_int):
        self.samples = samples
        self.word_to_int = word_to_int

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_seq = torch.LongTensor([self.word_to_int[word] for word in sample[:-1]])
        target_seq = torch.LongTensor([self.word_to_int[word] for word in sample[1:]])
        return input_seq, target_seq
```

### Training Loop

```python
def train(model, epochs, dataloader, criterion):
    model.train()
    for epoch in range(epochs):
        running_loss = 0
        for input_seq, target_seq in dataloader:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)

            outputs, _ = model(input_seq)
            loss = criterion(outputs, target_seq.view(-1))  # Flatten targets

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.detach().cpu().numpy()

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch} loss: {epoch_loss:.3f}")
```

### Generation Function

```python
def generate_text(model, start_string, num_words):
    model.eval()
    words = start_string.split()

    for _ in range(num_words):
        # Use last SEQUENCE_LENGTH words as context
        input_seq = torch.LongTensor(
            [word_to_int[word] for word in words[-SEQUENCE_LENGTH:]]
        ).unsqueeze(0).to(device)

        h, c = model.init_hidden(1)
        output, (h, c) = model(input_seq, (h, c))

        # Argmax selection (NOTE: can cause repetition - use sampling instead)
        next_token = output.argmax(1)[-1].item()
        words.append(int_to_word[next_token])

    return " ".join(words)
```

### Key Patterns to Adopt

1. **`batch_first=True`** for more intuitive tensor shapes
2. **`.contiguous().view(-1, hidden_size)`** before FC layer
3. **Flatten targets with `.view(-1)`** for CrossEntropyLoss
4. **Context window** in generation (last N words)

---

## Source 3: TensorFlow Tutorial (Architecture Patterns)

### Model Structure (Translate to PyTorch)

```python
# TensorFlow version
class MyModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

# PyTorch equivalent
class MyModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, rnn_units=1024):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, rnn_units, batch_first=True)
        self.dense = nn.Linear(rnn_units, vocab_size)

    def forward(self, x, states=None):
        x = self.embedding(x)
        output, states = self.gru(x, states)
        logits = self.dense(output)
        return logits, states
```

### Sequence Preparation Pattern

```python
def split_input_target(sequence):
    """Input is all but last, target is all but first (shifted by 1)"""
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

# Example: "Hello" -> input="Hell", target="ello"
seq_length = 100
```

### Temperature Sampling (Critical for Generation)

```python
def generate_one_step(model, input_ids, states, temperature=1.0):
    """
    Temperature controls randomness:
    - temp < 1.0: more deterministic/repetitive
    - temp > 1.0: more random/creative
    - temp = 1.0: default sampling
    """
    logits, states = model(input_ids, states)
    logits = logits[:, -1, :]  # Get last timestep

    # Apply temperature scaling
    logits = logits / temperature

    # CRITICAL: Sample, don't argmax (prevents loops)
    predicted_ids = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

    return predicted_ids, states
```

### Generation Loop with State Management

```python
def generate_text(model, start_string, num_generate, temperature=0.8):
    model.eval()

    # Encode start string
    input_ids = encode(start_string).unsqueeze(0)
    states = None
    result = list(start_string)

    for _ in range(num_generate):
        predicted_id, states = generate_one_step(model, input_ids, states, temperature)

        # Feed prediction back as next input
        input_ids = predicted_id

        # Decode and append
        result.append(decode(predicted_id.item()))

    return ''.join(result)
```

### Key Patterns to Adopt

1. **Temperature scaling**: `logits / temperature` before softmax
2. **Sampling vs argmax**: `torch.multinomial()` prevents repetition loops
3. **State management**: Pass hidden state between generation steps
4. **Mask unknown tokens**: Set logits to -inf for <UNK> tokens

---

## Source 4: MachineLearningMastery (Keras Patterns)

### Architecture Pattern

```python
# Keras style (from tutorial)
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=seq_length))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))

# PyTorch equivalent
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=50, hidden_dim=100, seq_length=50):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm1 = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]  # Take last timestep only
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

### Data Preparation

```python
# Tokenization with Keras Tokenizer equivalent in PyTorch
from collections import Counter

def build_vocab(text):
    words = text.lower().split()
    word_counts = Counter(words)
    vocab = ['<PAD>', '<UNK>'] + [w for w, c in word_counts.most_common()]
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for i, w in enumerate(vocab)}
    return word2idx, idx2word, len(vocab)

# Create sequences of fixed length
def create_sequences(text, word2idx, seq_length=50):
    words = text.lower().split()
    sequences = []
    for i in range(seq_length, len(words)):
        seq = words[i-seq_length:i+1]  # seq_length input + 1 target
        encoded = [word2idx.get(w, word2idx['<UNK>']) for w in seq]
        sequences.append(encoded)
    return sequences
```

### Key Patterns to Adopt

1. **Stacked LSTMs**: Two LSTM layers for more capacity
2. **Dense layer after LSTM**: Add non-linearity before output
3. **Take last timestep**: `x[:, -1, :]` when not using return_sequences
4. **Most common words first**: Better embedding for frequent words

---

## 1. Architecture Comparison

### Reference Architecture (Keras - lstm_train_embedding.py)

```
Input (word indices) → Embedding(1024) → Bidirectional(LSTM(128)) → Dropout(0.2) → Dense(vocab_size) → Softmax
```

| Component | Reference (Keras) | Current Project (PyTorch) |
|-----------|-------------------|---------------------------|
| Embedding | 1024-dim learned | 300-dim Word2Vec (frozen) |
| RNN Type | Bidirectional LSTM | Unidirectional LSTM |
| Hidden Units | 128 | 512 |
| Num Layers | 1 (bidirectional) | 2 |
| Dropout | 0.2 | 0.3 |
| Output | Dense + Softmax | Linear (logits) |

### Current Project Architecture

**Approach 1: Global Melody Conditioning** (`src/models.py:12-121`)
```
Word Embedding(300) + MIDI Global(128) → LSTM(512, 2 layers) → Dropout → Linear(vocab_size)
```

**Approach 2: Attention over Melody** (`src/models.py:183-319`)
```
Word Embedding(300) + Attention(MIDI Temporal) → LSTM(512, 2 layers) → Dropout → Linear(vocab_size)
```

---

## 2. Key Reference Implementation Details

### From lstm_train_embedding.py (GitHub)

```python
# Sequence preparation
SEQUENCE_LEN = 10      # Context window: 10 words
STEP = 1               # Sliding stride
MIN_WORD_FREQUENCY = 10  # Filter rare words

# Model hyperparameters
embedding_dim = 1024   # Learned embedding size
lstm_units = 128       # LSTM hidden dimension
dropout = 0.2          # Regularization

# Training
BATCH_SIZE = 32
EPOCHS = 100
optimizer = 'adam'
loss = 'sparse_categorical_crossentropy'
early_stopping_patience = 20
```

### Sequence Preparation (Reference)
```python
# Create overlapping sequences with step=1
for i in range(0, len(words) - SEQUENCE_LEN, STEP):
    sentence = words[i:i + SEQUENCE_LEN]  # Input: 10 words
    next_word = words[i + SEQUENCE_LEN]    # Target: next word

    # Filter sequences with rare words
    if all words in sentence have freq >= MIN_WORD_FREQUENCY:
        sentences.append(sentence)
        next_words.append(next_word)
```

### Word-to-Index Mapping (Reference)
```python
word_indices = {word: idx for idx, word in enumerate(unique_words)}
indices_word = {idx: word for idx, word in enumerate(unique_words)}

# Encode sequence
X = np.array([[word_indices[w] for w in sentence] for sentence in sentences])
y = np.array([word_indices[w] for w in next_words])
```

---

## 3. Implementation Plan

### Phase 1: Data Preparation Enhancements

**File: `src/data_loader.py`**

#### 1.1 Update Vocabulary Class (lines 27-101)

Current implementation already handles:
- [x] Special tokens: `<PAD>`, `<UNK>`, `<SOS>`, `<EOS>`, `<NEWLINE>`
- [x] Word-to-index mapping
- [x] Tokenization (lowercase, whitespace split)

**Enhancement needed:**
- Add `MIN_WORD_FREQUENCY` filtering like reference (optional, improves training)

```python
# Add to config.py
MIN_WORD_FREQUENCY = 10  # Filter words appearing less than this

# In Vocabulary class, add method:
def filter_vocabulary(self, min_freq: int):
    """Remove words with frequency below threshold."""
    filtered = {w: idx for w, idx in self.word2idx.items()
                if self.word_count.get(w, 0) >= min_freq
                or w in SPECIAL_TOKENS}
    # Rebuild indices
```

#### 1.2 Sequence Preparation (lines 268-298)

Current implementation:
- Input: `[SOS] + lyrics_indices`
- Target: `lyrics_indices + [EOS]`
- Max length: 200 tokens (truncation)

**This matches the shifted-by-one approach from reference.**

---

### Phase 2: Model Architecture Updates

**File: `src/models.py`**

#### 2.1 Consider Bidirectional LSTM Option

Reference uses `Bidirectional(LSTM(128))` which effectively doubles output dimension.

**Add to config.py:**
```python
BIDIRECTIONAL = True  # Option to use bidirectional LSTM
```

**Update LyricsLSTMGlobal (lines 27-121):**
```python
# In __init__:
self.lstm = nn.LSTM(
    input_size=lstm_input_dim,
    hidden_size=hidden_dim,
    num_layers=num_layers,
    batch_first=True,
    dropout=dropout if num_layers > 1 else 0,
    bidirectional=config.BIDIRECTIONAL  # Add this
)

# Adjust output layer
lstm_output_dim = hidden_dim * 2 if config.BIDIRECTIONAL else hidden_dim
self.fc_out = nn.Linear(lstm_output_dim, vocab_size)
```

#### 2.2 Embedding Layer Options

Current: Word2Vec (300-dim, frozen)
Reference: Learned embedding (1024-dim)

**Add config option:**
```python
# config.py
USE_PRETRAINED_EMBEDDINGS = True  # False = learned like reference
EMBEDDING_DIM = 300 if USE_PRETRAINED_EMBEDDINGS else 1024
FREEZE_EMBEDDINGS = True
```

**Update model:**
```python
# Option for learned embeddings
if not config.USE_PRETRAINED_EMBEDDINGS:
    self.embedding = nn.Embedding(vocab_size, config.EMBEDDING_DIM)
else:
    # Current Word2Vec approach
    self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
```

---

### Phase 3: Training Configuration

**File: `src/train.py` and `src/config.py`**

#### 3.1 Update Hyperparameters (config.py)

```python
# Current values (keep these - they're appropriate)
BATCH_SIZE = 32          # Same as reference
LEARNING_RATE = 0.001    # Same as reference (Adam default)
NUM_EPOCHS = 50          # Reference uses 100 with early stopping
GRADIENT_CLIP = 5.0      # Good for LSTM stability

# Add early stopping (reference uses patience=20)
EARLY_STOPPING_PATIENCE = 10
```

#### 3.2 Training Loop Enhancements (train.py)

Current implementation already has:
- [x] Adam optimizer
- [x] Learning rate scheduler (ReduceLROnPlateau)
- [x] TensorBoard logging
- [x] Best model checkpointing
- [x] Structure-aware loss with <NEWLINE> boosting

**Add early stopping:**
```python
# In train_model():
early_stopping_counter = 0

for epoch in range(num_epochs):
    # ... training ...

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stopping_counter = 0
        # save checkpoint
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}")
            break
```

---

### Phase 4: Generation Strategy

**File: `src/generate.py`**

#### 4.1 Current Implementation (Already Matches Reference Approach)

```python
# Current: Top-k sampling with temperature
def sample_next_word(logits, temperature=0.8, top_k=50):
    logits = logits / temperature
    # Top-k filtering
    top_k_values, top_k_indices = torch.topk(logits, top_k)
    # ... softmax and sample
    sampled_idx = torch.multinomial(probs, num_samples=1)
```

This is appropriate for creative text generation.

#### 4.2 Insights from TensorFlow Tutorial

The TensorFlow tutorial emphasizes:

```python
# Temperature controls randomness:
# - Lower temp (0.5) = more predictable, repetitive
# - Higher temp (1.5) = more creative, potentially nonsensical
# - Sweet spot: 0.7-1.0 for coherent creativity

# CRITICAL: Use sampling, not argmax
# Argmax causes repetition loops
predicted_ids = tf.random.categorical(logits, num_samples=1)  # Good
# NOT: predicted_ids = tf.argmax(logits)  # Bad - causes loops
```

#### 4.3 State Management for Generation

From TensorFlow tutorial - maintain RNN state across generation:

```python
# Current project correctly implements this:
def generate_lyrics(...):
    hidden = None  # Initial state
    for _ in range(max_length):
        output, hidden = model(current_word, midi_input, hidden)  # Pass state
        # ... sample next word
        current_word = next_word  # Feed output back
```

---

### Phase 5: MIDI Integration (Project-Specific)

**File: `src/midi_features.py`**

This is unique to your project. The reference doesn't include melody conditioning.

#### 5.1 Global Features (128-dim) - Already Implemented

| Feature Category | Dimensions |
|------------------|------------|
| Tempo stats | 7 |
| Piano roll | 32 |
| Instruments | 18 |
| Note statistics | 20 |
| Chroma | 12 |
| **Total** | **89 → padded to 128** |

#### 5.2 Temporal Features (50 x 32) - Already Implemented

For attention model - captures melody evolution over time.

---

## 4. Summary: Files to Modify

| File | Changes | Priority |
|------|---------|----------|
| `src/config.py` | Add BIDIRECTIONAL, EARLY_STOPPING_PATIENCE, optional MIN_WORD_FREQUENCY | Medium |
| `src/models.py` | Add bidirectional option to LSTM | Low |
| `src/train.py` | Add early stopping logic | Medium |
| `src/data_loader.py` | Optional: word frequency filtering | Low |
| `src/generate.py` | No changes needed | - |
| `src/losses.py` | No changes needed | - |

---

## 5. Key Differences Summary

| Aspect | Reference | Your Project | Recommendation |
|--------|-----------|--------------|----------------|
| Embedding | 1024-dim learned | 300-dim Word2Vec | Keep Word2Vec - better for small datasets |
| LSTM Direction | Bidirectional | Unidirectional | Keep unidirectional for generation (bidirectional can't do autoregressive) |
| Hidden Units | 128 | 512 | Your value is fine - more capacity |
| Melody Conditioning | None | Global + Attention | Unique feature - keep it |
| Loss | Sparse CE | Structure-aware CE | Your version is better for lyrics |
| Generation | Various sampling | Top-k + temperature | Good approach |

---

## 6. Implementation Priority

### Must Have (Core Functionality)
1. [x] Word-level tokenization - **Already done**
2. [x] LSTM architecture - **Already done**
3. [x] Word2Vec embeddings - **Already done**
4. [x] MIDI feature extraction - **Already done**
5. [x] Training loop with Adam - **Already done**
6. [x] Top-k sampling generation - **Already done**

### Should Have (Improvements)
1. [ ] Early stopping in training
2. [ ] Better training/validation logging

### Nice to Have (Enhancements)
1. [ ] Bidirectional option (though not recommended for generation)
2. [ ] Word frequency filtering
3. [ ] Learned embeddings option

---

## 7. PyTorch vs Keras Translation Reference

### Keras (Reference)
```python
model = Sequential()
model.add(Embedding(len(words), 1024))
model.add(Bidirectional(LSTM(128)))
model.add(Dropout(0.2))
model.add(Dense(len(words), activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
```

### PyTorch (Your Project)
```python
class LyricsLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                           num_layers=2, batch_first=True, dropout=0.3)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embedding(x)
        out, hidden = self.lstm(emb, hidden)
        logits = self.fc_out(out)
        return logits, hidden

# Training
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

---

## 8. Specific Improvements from Sources

### From FloydHub (PyTorch Reference)

**Current code (`src/models.py`) vs FloydHub pattern:**

```python
# CURRENT: Dropout only between LSTM layers
self.lstm = nn.LSTM(..., dropout=dropout if num_layers > 1 else 0)
self.dropout = nn.Dropout(dropout)  # Only on output

# FLOYHUB PATTERN: Dropout on embeddings AND output
def forward(self, input, hidden):
    emb = self.drop(self.encoder(input))  # <-- ADD: Dropout on embeddings
    output, hidden = self.rnn(emb, hidden)
    output = self.drop(output)  # Dropout on output
    ...
```

**Recommended change to `src/models.py:74-115`:**
```python
def forward(self, input_seq, midi_features, hidden=None, lengths=None):
    batch_size, seq_len = input_seq.shape

    # Get word embeddings with dropout (FloydHub pattern)
    embeddings = self.dropout(self.embedding(input_seq))  # ADD DROPOUT HERE

    # ... rest of forward pass
```

### From DebuggerCafe (PyTorch Tutorial)

**Already implemented correctly:**
- [x] `batch_first=True` in LSTM
- [x] Hidden state initialization with zeros
- [x] Dataset class with `__getitem__` returning (input, target)

**Could improve generation (`src/generate.py`):**
```python
# CURRENT: Reinitializes hidden every step
hidden = None
for _ in range(max_length - 1):
    output, hidden = model(current_word, midi_input, hidden)  # Good - passes hidden

# This is correct! Hidden state IS preserved between steps.
```

### From TensorFlow Tutorial

**Temperature sampling - Already implemented correctly in `src/generate.py:14-50`:**
```python
# CURRENT (correct):
logits = logits / temperature  # Temperature scaling
probs = F.softmax(logits, dim=-1)
sampled_idx = torch.multinomial(probs, num_samples=1)  # Sampling, not argmax
```

**Could add: UNK token masking:**
```python
# Add to sample_next_word() in src/generate.py
def sample_next_word(logits, temperature, top_k, unk_idx=None):
    logits = logits / temperature

    # Mask UNK token to prevent generating unknown words
    if unk_idx is not None:
        logits[unk_idx] = float('-inf')

    # ... rest of function
```

### From MachineLearningMastery

**Vocab building - Already implemented in `src/data_loader.py:27-101`:**
- [x] Special tokens first (`<PAD>`, `<UNK>`, etc.)
- [x] Word frequency counting
- [x] `word2idx` and `idx2word` dictionaries

**Could add: Sort by frequency:**
```python
# In Vocabulary class, optionally sort by frequency
vocab = ['<PAD>', '<UNK>', '<SOS>', '<EOS>', '<NEWLINE>']
vocab += [w for w, c in self.word_count.most_common()]  # Most common first
```

---

## 9. Summary of Recommended Changes

| File | Change | Source | Priority |
|------|--------|--------|----------|
| `src/models.py` | Add dropout on embeddings in forward() | FloydHub | High |
| `src/generate.py` | Add UNK token masking | TensorFlow | Medium |
| `src/train.py` | Add early stopping | FloydHub | Medium |
| `src/config.py` | Add EARLY_STOPPING_PATIENCE | All | Medium |
| `src/models.py` | Optional: weight initialization [-0.1, 0.1] | FloydHub | Low |
| `src/data_loader.py` | Optional: sort vocab by frequency | MLMastery | Low |

---

## 10. Two Melody Integration Approaches (Assignment Requirement)

The assignment requires **two meaningfully different approaches** for integrating melody information.

### Approach 1: Global Melody Conditioning

**File:** `src/models.py` - `LyricsLSTMGlobal` class (lines 12-121)

**How it works:**
```
MIDI File → Extract 128-dim global feature vector → Concatenate with word embedding at EACH timestep

Input to LSTM = [Word2Vec(300) || MIDI_global(128)] = 428 dimensions
```

**Diagram:**
```
                    ┌─────────────┐
MIDI ──────────────►│ 128-dim     │──┐
                    │ Global Vec  │  │
                    └─────────────┘  │
                                     ▼
Word[t] ──► Embedding(300) ──────► Concat ──► LSTM ──► Dense ──► P(word[t+1])
                                     ▲
                                     │
                    (same MIDI vector repeated for all timesteps)
```

**Characteristics:**
- Simple concatenation
- Same melody representation for entire song
- Loses temporal alignment between lyrics and melody
- Faster training (no attention computation)

### Approach 2: Temporal Melody with Attention

**File:** `src/models.py` - `LyricsLSTMAttention` class (lines 183-319)

**How it works:**
```
MIDI File → Extract temporal features (50 frames × 32 dims) → Attention mechanism selects relevant frames

At each timestep:
  Query = LSTM hidden state
  Keys/Values = MIDI temporal sequence
  Context = Weighted sum of MIDI frames based on attention
```

**Diagram:**
```
                    ┌─────────────────────────┐
MIDI ──────────────►│ 50 frames × 32 dims     │
                    │ Temporal Features       │
                    └───────────┬─────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │     Attention           │◄─── Query from LSTM hidden[t]
                    │  (learns which frames   │
                    │   matter for word[t])   │
                    └───────────┬─────────────┘
                                │
                                ▼
Word[t] ──► Embedding(300) ──► Concat ──► LSTM ──► Dense ──► P(word[t+1])
                                ▲
                                │
                    Context(32) = weighted MIDI frames
```

**Characteristics:**
- Dynamic melody selection per word
- Captures temporal alignment (verse vs chorus melody)
- More parameters, slower training
- Potentially better melody-lyrics coherence

### Comparison Table

| Aspect | Global | Attention |
|--------|--------|-----------|
| MIDI representation | Single 128-dim vector | 50 × 32 temporal sequence |
| Input dimension | 428 (300 + 128) | 332 (300 + 32 context) |
| Temporal alignment | ❌ None | ✅ Dynamic per word |
| Computational cost | Lower | Higher |
| Implementation | `LyricsLSTMGlobal` | `LyricsLSTMAttention` |

### Why These Are "Meaningfully Different"

1. **Information granularity**: Global uses song-level summary; Attention uses frame-level detail
2. **Architecture difference**: Global is feedforward concat; Attention adds query-key-value mechanism
3. **Hypothesis tested**: Does temporal melody alignment improve lyric generation?

---

## 11. Conclusion

Your current implementation already captures the essential architecture from the reference, with improvements:

1. **Better embeddings**: Word2Vec provides semantic meaning vs. learned embeddings
2. **Richer model**: More hidden units (512 vs 128) and layers (2 vs 1)
3. **Melody conditioning**: Unique feature not in reference
4. **Structure-aware loss**: Teaches proper lyric formatting

**Recommended next steps:**
1. Add early stopping to `train.py`
2. Run training and evaluate generated lyrics
3. Fine-tune hyperparameters based on results
