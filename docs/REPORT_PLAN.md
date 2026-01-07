# Assignment 3 Report Outline

**Due Date:** 14.01.2026
**Format:** PDF (max 6 pages text, figures in appendix)
**Font:** Calibri 12pt, 2.5cm margins

---

## 1. Introduction (~0.5 page)

- Task: Generate song lyrics conditioned on MIDI melody
- Challenge: Integrating different information sources (text + audio)
- Overview of two implemented approaches

---

## 2. Model Architecture (~1.5 pages)

### 2.1 Base Architecture
- LSTM-based language model
- Word2Vec embeddings (300-dim, pre-trained)
- 2-layer LSTM with 512 hidden units
- Dropout regularization (0.3)

### 2.2 Approach 1: Global Melody Conditioning
- Extract single 128-dim vector from entire MIDI file
- Concatenate with word embedding at each timestep
- Input to LSTM: [word_emb (300) + midi_global (128)] = 428-dim
- Simple but loses temporal alignment

### 2.3 Approach 2: Temporal Melody with Attention
- Extract temporal MIDI features: 50 frames x 32 dimensions
- Attention mechanism computes weighted context at each step
- Query: LSTM hidden state
- Keys/Values: Temporal MIDI sequence
- Input to LSTM: [word_emb (300) + attention_context (32)] = 332-dim
- Allows dynamic focus on melody parts

### 2.4 Architecture Diagram
[Include figure showing both approaches side-by-side]

---

## 3. MIDI Feature Extraction (~1 page)

### 3.1 Using Pretty_MIDI
- Library for parsing .mid files
- Access to notes, instruments, tempo, timing

### 3.2 Global Features (128-dim)

| Category | Features | Dimensions |
|----------|----------|------------|
| Tempo | mean, std, min, max, duration, time sig | 7 |
| Piano Roll | octave activity, velocity stats, note density | 32 |
| Instruments | family distribution, count, drums | 18 |
| Notes | pitch/duration/velocity stats, intervals | 20 |
| Chroma | pitch class distribution | 12 |
| **Total** | | **128** |

### 3.3 Temporal Features (50 x 32)
- Divide song into 50 time frames
- Per-frame: octave activity, velocity, note count
- Captures melody evolution over time

---

## 4. Training Details (~1 page)

### 4.1 Data Preparation
- 599 training songs, 4 test songs
- Train/Validation split: 90/10
- Vocabulary: word-level tokenization
- Max sequence length: 200 tokens
- Special tokens: `<SOS>`, `<EOS>`, `<PAD>`, `<UNK>`, `<NEWLINE>`

### 4.2 Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 32 |
| Epochs | 50 |
| Gradient clipping | 5.0 (norm) |
| Teacher forcing ratio | 0.5 |

### 4.3 Loss Function
- CrossEntropyLoss
- Padding tokens masked

### 4.4 Training Curves
[Reference to Appendix - TensorBoard graphs]
- Training loss vs epochs (both models)
- Validation loss vs epochs (both models)

---

## 5. Generation Strategy (~0.5 page)

### 5.1 Sampling Method
- **Top-k sampling** (k=50): Sample from top 50 probable words
- **Temperature** (T=0.8): Controls randomness
- Probabilistic, not deterministic (as required)

### 5.2 Generation Parameters
- Max length: 200 words
- Max words per line: 10
- Stop on `<EOS>` token

### 5.3 Line Structure
- `<NEWLINE>` token for line breaks
- Creates song-like formatting

---

## 6. Results and Analysis (~1.5 pages)

### 6.1 Test Setup
- 4 test melodies
- 3 starting words per melody: `"love"`, `"the"`, `"i"`
- Generate with both Global and Attention models

### 6.2 Generated Lyrics Table

| Melody | Start Word | Global Model Output | Attention Model Output |
|--------|------------|---------------------|------------------------|
| Song 1 | "love" | [output] | [output] |
| Song 1 | "the" | [output] | [output] |
| Song 1 | "i" | [output] | [output] |
| Song 2 | "love" | [output] | [output] |
| ... | ... | ... | ... |

[Full lyrics in Appendix]

### 6.3 Analysis: Effect of Starting Word
- How does the first word influence the generated content?
- Thematic consistency with starting word?
- Compare across different melodies

### 6.4 Analysis: Effect of Melody
- Do different melodies produce different lyric styles?
- Tempo/energy correlation with lyric mood?
- Compare same starting word across different melodies

### 6.5 Comparison: Global vs Attention
- Quality of generated lyrics
- Coherence and structure
- Does attention improve melody-lyrics alignment?
- Observations on attention weights (if visualized)

---

## 7. Conclusion (~0.25 page)

- Summary of findings
- Which approach performed better and why
- Limitations and future improvements

---

## Appendix (No page limit)

### A. TensorBoard Graphs
- Figure A1: Training loss - Global model
- Figure A2: Validation loss - Global model
- Figure A3: Training loss - Attention model
- Figure A4: Validation loss - Attention model

### B. Full Generated Lyrics
- All outputs for 4 melodies x 3 starting words x 2 models

### C. Additional Figures
- Architecture diagrams
- Attention weight visualizations (optional)
- MIDI feature distributions

---

## Checklist Before Submission

- [ ] Train both models to completion
- [ ] Generate TensorBoard graphs
- [ ] Generate lyrics for all test melodies with 3 starting words
- [ ] Write analysis of results
- [ ] Create architecture diagrams
- [ ] Compile report (max 6 pages text)
- [ ] Export to PDF
- [ ] Zip code + report together
