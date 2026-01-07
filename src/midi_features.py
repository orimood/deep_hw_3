"""
MIDI feature extraction using Pretty_MIDI.
Extracts both global and temporal features for melody conditioning.
"""

import numpy as np
import pretty_midi
from typing import Tuple, Optional

import config


def extract_global_features(midi_path: str) -> np.ndarray:
    """
    Extract global (song-level) features from a MIDI file.
    Returns a fixed 128-dim feature vector.

    Features include:
    - Tempo statistics (7 features)
    - Piano roll statistics (32 features)
    - Instrument features (18 features)
    - Note statistics (20 features)
    - Chroma/pitch class (12 features)
    - Padding to 128 total
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"Error loading MIDI file {midi_path}: {e}")
        return np.zeros(config.MIDI_GLOBAL_DIM, dtype=np.float32)

    features = []

    # 1. Tempo features (7)
    features.extend(_extract_tempo_features(midi_data))

    # 2. Piano roll features (32)
    features.extend(_extract_piano_roll_features(midi_data))

    # 3. Instrument features (18)
    features.extend(_extract_instrument_features(midi_data))

    # 4. Note statistics (20)
    features.extend(_extract_note_features(midi_data))

    # 5. Chroma features (12)
    features.extend(_extract_chroma_features(midi_data))

    # Convert to numpy and pad/truncate to fixed size
    features = np.array(features, dtype=np.float32)

    if len(features) < config.MIDI_GLOBAL_DIM:
        features = np.pad(features, (0, config.MIDI_GLOBAL_DIM - len(features)))
    else:
        features = features[:config.MIDI_GLOBAL_DIM]

    # Normalize to [0, 1]
    features = _normalize_features(features)

    return features


def extract_temporal_features(midi_path: str, num_frames: int = config.MIDI_TEMPORAL_FRAMES) -> np.ndarray:
    """
    Extract temporal (frame-level) features from a MIDI file.
    Returns features of shape (num_frames, frame_dim).

    Used for the Attention-based melody integration approach.
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
    except Exception:
        return np.zeros((num_frames, config.MIDI_FRAME_DIM), dtype=np.float32)

    total_time = midi_data.get_end_time()
    if total_time == 0:
        return np.zeros((num_frames, config.MIDI_FRAME_DIM), dtype=np.float32)

    frame_duration = total_time / num_frames
    frame_features = []

    for i in range(num_frames):
        start_time = i * frame_duration
        end_time = (i + 1) * frame_duration

        features = []

        # Notes active in this frame
        active_notes = []
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                if note.start < end_time and note.end > start_time:
                    active_notes.append(note)

        # Note features (4)
        if active_notes:
            pitches = [n.pitch for n in active_notes]
            velocities = [n.velocity for n in active_notes]
            features.append(np.mean(pitches) / 127)
            features.append(np.std(pitches) / 127 if len(pitches) > 1 else 0)
            features.append(np.mean(velocities) / 127)
            features.append(len(active_notes) / 20)  # Normalized count
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])

        # Chroma for this frame (12)
        chroma = _get_frame_chroma(midi_data, start_time, end_time)
        features.extend(chroma)

        # Tempo at this frame (1)
        tempo = _get_tempo_at_time(midi_data, start_time)
        features.append(tempo / 200)  # Normalized

        # Pad to frame_dim
        while len(features) < config.MIDI_FRAME_DIM:
            features.append(0.0)
        features = features[:config.MIDI_FRAME_DIM]

        frame_features.append(features)

    return np.array(frame_features, dtype=np.float32)


def _extract_tempo_features(midi_data: pretty_midi.PrettyMIDI) -> list:
    """Extract tempo-related features (7 total)."""
    features = []

    tempo_changes = midi_data.get_tempo_changes()
    tempos = tempo_changes[1] if len(tempo_changes[1]) > 0 else [120.0]

    features.append(np.mean(tempos))
    features.append(np.std(tempos) if len(tempos) > 1 else 0.0)
    features.append(np.min(tempos))
    features.append(np.max(tempos))
    features.append(midi_data.get_end_time())

    # Time signature
    time_sigs = midi_data.time_signature_changes
    if time_sigs:
        features.append(time_sigs[0].numerator)
        features.append(time_sigs[0].denominator)
    else:
        features.extend([4, 4])

    return features


def _extract_piano_roll_features(midi_data: pretty_midi.PrettyMIDI) -> list:
    """Extract piano roll statistics (32 total)."""
    features = []

    try:
        piano_roll = midi_data.get_piano_roll(fs=config.MIDI_PIANO_ROLL_FS)
        if piano_roll.size == 0:
            return [0.0] * 32

        # Truncate if too long
        max_frames = 1000
        if piano_roll.shape[1] > max_frames:
            piano_roll = piano_roll[:, :max_frames]

        # Octave activity (11)
        for octave in range(11):
            start_pitch = octave * 12
            end_pitch = min((octave + 1) * 12, 128)
            octave_sum = np.sum(piano_roll[start_pitch:end_pitch, :])
            features.append(octave_sum)

        # Normalize octave activity
        total = sum(features[-11:])
        if total > 0:
            features[-11:] = [x / total for x in features[-11:]]

        # Overall statistics (6)
        features.append(np.mean(piano_roll))
        features.append(np.std(piano_roll))
        features.append(np.sum(piano_roll > 0) / piano_roll.size)

        time_activity = np.sum(piano_roll, axis=0)
        if len(time_activity) > 0:
            features.append(np.mean(time_activity))
            features.append(np.std(time_activity))
            features.append(np.max(time_activity))

            # Section activity (3)
            n_frames = len(time_activity)
            third = n_frames // 3
            if third > 0:
                features.append(np.mean(time_activity[:third]))
                features.append(np.mean(time_activity[third:2*third]))
                features.append(np.mean(time_activity[2*third:]))
            else:
                features.extend([0.0, 0.0, 0.0])
        else:
            features.extend([0.0] * 6)

    except Exception:
        features = [0.0] * 32

    # Ensure exactly 32 features
    while len(features) < 32:
        features.append(0.0)
    return features[:32]


def _extract_instrument_features(midi_data: pretty_midi.PrettyMIDI) -> list:
    """Extract instrument distribution features (18 total)."""
    features = []

    # Instrument family counts (16)
    instrument_counts = [0] * 16
    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            family = instrument.program // 8
            if family < 16:
                instrument_counts[family] += 1

    # Normalize
    total = sum(instrument_counts)
    if total > 0:
        instrument_counts = [x / total for x in instrument_counts]
    features.extend(instrument_counts)

    # Total instruments and drums flag (2)
    features.append(len(midi_data.instruments))
    features.append(1.0 if any(i.is_drum for i in midi_data.instruments) else 0.0)

    return features


def _extract_note_features(midi_data: pretty_midi.PrettyMIDI) -> list:
    """Extract note-level statistics (20 total)."""
    features = []

    all_notes = []
    for instrument in midi_data.instruments:
        all_notes.extend(instrument.notes)

    if not all_notes:
        return [0.0] * 20

    # Pitch statistics (5)
    pitches = [note.pitch for note in all_notes]
    features.append(np.mean(pitches))
    features.append(np.std(pitches))
    features.append(np.min(pitches))
    features.append(np.max(pitches))
    features.append(np.max(pitches) - np.min(pitches))

    # Duration statistics (4)
    durations = [note.end - note.start for note in all_notes]
    features.append(np.mean(durations))
    features.append(np.std(durations))
    features.append(np.min(durations))
    features.append(np.max(durations))

    # Velocity statistics (4)
    velocities = [note.velocity for note in all_notes]
    features.append(np.mean(velocities))
    features.append(np.std(velocities))
    features.append(np.min(velocities))
    features.append(np.max(velocities))

    # Density features (2)
    features.append(len(all_notes))
    total_time = midi_data.get_end_time()
    features.append(len(all_notes) / max(total_time, 1))

    # Inter-onset intervals (2)
    starts = sorted([note.start for note in all_notes])
    if len(starts) > 1:
        iois = np.diff(starts)
        features.append(np.mean(iois))
        features.append(np.std(iois))
    else:
        features.extend([0.0, 0.0])

    # Pitch intervals (2)
    sorted_notes = sorted(all_notes, key=lambda x: x.start)
    if len(sorted_notes) > 1:
        intervals = [sorted_notes[i+1].pitch - sorted_notes[i].pitch
                    for i in range(len(sorted_notes)-1)]
        features.append(np.mean(np.abs(intervals)))
        features.append(np.std(intervals))
    else:
        features.extend([0.0, 0.0])

    return features[:20]


def _extract_chroma_features(midi_data: pretty_midi.PrettyMIDI) -> list:
    """Extract chroma/pitch class distribution (12 total)."""
    try:
        chroma = midi_data.get_chroma(fs=config.MIDI_PIANO_ROLL_FS)
        if chroma.size == 0:
            return [0.0] * 12

        avg_chroma = np.mean(chroma, axis=1)
        total = np.sum(avg_chroma)
        if total > 0:
            avg_chroma = avg_chroma / total

        return avg_chroma.tolist()
    except Exception:
        return [0.0] * 12


def _get_frame_chroma(midi_data: pretty_midi.PrettyMIDI, start_time: float, end_time: float) -> list:
    """Get chroma features for a specific time frame."""
    try:
        piano_roll = midi_data.get_piano_roll(fs=config.MIDI_PIANO_ROLL_FS)
        start_idx = int(start_time * config.MIDI_PIANO_ROLL_FS)
        end_idx = int(end_time * config.MIDI_PIANO_ROLL_FS)

        if start_idx < piano_roll.shape[1]:
            end_idx = min(end_idx, piano_roll.shape[1])
            frame_roll = piano_roll[:, start_idx:end_idx]

            chroma = np.zeros(12)
            for pitch in range(128):
                chroma[pitch % 12] += np.sum(frame_roll[pitch, :])

            total = np.sum(chroma)
            if total > 0:
                chroma = chroma / total
            return chroma.tolist()
    except Exception:
        pass

    return [0.0] * 12


def _get_tempo_at_time(midi_data: pretty_midi.PrettyMIDI, time: float) -> float:
    """Get tempo at a specific time."""
    tempos = midi_data.get_tempo_changes()
    current_tempo = 120.0
    for t, tempo in zip(tempos[0], tempos[1]):
        if t <= time:
            current_tempo = tempo
    return current_tempo


def _normalize_features(features: np.ndarray) -> np.ndarray:
    """Normalize features to [0, 1] range."""
    min_val = np.min(features)
    max_val = np.max(features)
    if max_val - min_val > 0:
        features = (features - min_val) / (max_val - min_val)
    return features
