"""MIDI feature extraction using Pretty_Midi."""

import numpy as np
import pretty_midi
from typing import Dict, List, Tuple, Optional
from . import config


def extract_midi_features(midi_path: str) -> np.ndarray:
    """
    Extract comprehensive features from a MIDI file.

    Returns a fixed-size feature vector representing the melody.
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"Error loading MIDI file {midi_path}: {e}")
        return np.zeros(config.MIDI_FEATURE_DIM)

    features = []

    # 1. Global tempo and timing features
    tempo_features = extract_tempo_features(midi_data)
    features.extend(tempo_features)

    # 2. Piano roll statistics
    piano_roll_features = extract_piano_roll_features(midi_data)
    features.extend(piano_roll_features)

    # 3. Instrument features
    instrument_features = extract_instrument_features(midi_data)
    features.extend(instrument_features)

    # 4. Note statistics
    note_features = extract_note_features(midi_data)
    features.extend(note_features)

    # 5. Chroma features (pitch class distribution)
    chroma_features = extract_chroma_features(midi_data)
    features.extend(chroma_features)

    # Pad or truncate to fixed size
    features = np.array(features, dtype=np.float32)

    if len(features) < config.MIDI_FEATURE_DIM:
        features = np.pad(features, (0, config.MIDI_FEATURE_DIM - len(features)))
    elif len(features) > config.MIDI_FEATURE_DIM:
        features = features[:config.MIDI_FEATURE_DIM]

    # Normalize features
    features = normalize_features(features)

    return features


def extract_tempo_features(midi_data: pretty_midi.PrettyMIDI) -> List[float]:
    """Extract tempo-related features."""
    features = []

    # Get tempo changes
    tempo_changes = midi_data.get_tempo_changes()
    tempos = tempo_changes[1] if len(tempo_changes[1]) > 0 else [120.0]

    features.append(np.mean(tempos))  # Average tempo
    features.append(np.std(tempos) if len(tempos) > 1 else 0.0)  # Tempo variation
    features.append(np.min(tempos))  # Min tempo
    features.append(np.max(tempos))  # Max tempo

    # Duration
    features.append(midi_data.get_end_time())

    # Time signature info
    time_sigs = midi_data.time_signature_changes
    if time_sigs:
        features.append(time_sigs[0].numerator)
        features.append(time_sigs[0].denominator)
    else:
        features.extend([4, 4])  # Default 4/4

    return features


def extract_piano_roll_features(midi_data: pretty_midi.PrettyMIDI) -> List[float]:
    """Extract features from the piano roll representation."""
    features = []

    try:
        # Get piano roll (128 pitches x time)
        piano_roll = midi_data.get_piano_roll(fs=config.MIDI_PIANO_ROLL_FS)

        if piano_roll.size == 0:
            return [0.0] * 32

        # Truncate to max length
        if piano_roll.shape[1] > config.MIDI_MAX_LENGTH:
            piano_roll = piano_roll[:, :config.MIDI_MAX_LENGTH]

        # Statistics per pitch (128 dimensions -> reduce to manageable size)
        # Group pitches into octaves (12 pitches per octave, ~10 octaves)
        octave_activity = []
        for octave in range(11):
            start_pitch = octave * 12
            end_pitch = min((octave + 1) * 12, 128)
            octave_sum = np.sum(piano_roll[start_pitch:end_pitch, :])
            octave_activity.append(octave_sum)

        # Normalize octave activity
        total = sum(octave_activity)
        if total > 0:
            octave_activity = [x / total for x in octave_activity]
        features.extend(octave_activity)

        # Overall statistics
        features.append(np.mean(piano_roll))  # Average velocity
        features.append(np.std(piano_roll))   # Velocity variation
        features.append(np.sum(piano_roll > 0) / piano_roll.size)  # Note density

        # Temporal statistics - activity over time
        time_activity = np.sum(piano_roll, axis=0)
        if len(time_activity) > 0:
            features.append(np.mean(time_activity))
            features.append(np.std(time_activity))
            features.append(np.max(time_activity))

            # Activity in different sections (beginning, middle, end)
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

    except Exception as e:
        features = [0.0] * 32

    # Ensure fixed size
    while len(features) < 32:
        features.append(0.0)

    return features[:32]


def extract_instrument_features(midi_data: pretty_midi.PrettyMIDI) -> List[float]:
    """Extract instrument-related features."""
    features = []

    # Count instruments by type
    instrument_counts = [0] * 16  # 16 instrument families

    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            program = instrument.program
            family = program // 8  # 8 instruments per family
            if family < 16:
                instrument_counts[family] += 1

    # Normalize
    total = sum(instrument_counts)
    if total > 0:
        instrument_counts = [x / total for x in instrument_counts]

    features.extend(instrument_counts)

    # Number of instruments
    features.append(len(midi_data.instruments))

    # Has drums?
    has_drums = any(inst.is_drum for inst in midi_data.instruments)
    features.append(1.0 if has_drums else 0.0)

    return features


def extract_note_features(midi_data: pretty_midi.PrettyMIDI) -> List[float]:
    """Extract note-level statistics."""
    features = []

    all_notes = []
    for instrument in midi_data.instruments:
        all_notes.extend(instrument.notes)

    if not all_notes:
        return [0.0] * 20

    # Pitch statistics
    pitches = [note.pitch for note in all_notes]
    features.append(np.mean(pitches))
    features.append(np.std(pitches))
    features.append(np.min(pitches))
    features.append(np.max(pitches))
    features.append(np.max(pitches) - np.min(pitches))  # Range

    # Duration statistics
    durations = [note.end - note.start for note in all_notes]
    features.append(np.mean(durations))
    features.append(np.std(durations))
    features.append(np.min(durations))
    features.append(np.max(durations))

    # Velocity statistics
    velocities = [note.velocity for note in all_notes]
    features.append(np.mean(velocities))
    features.append(np.std(velocities))
    features.append(np.min(velocities))
    features.append(np.max(velocities))

    # Note count and density
    features.append(len(all_notes))
    total_time = midi_data.get_end_time()
    features.append(len(all_notes) / max(total_time, 1))  # Notes per second

    # Inter-onset intervals
    starts = sorted([note.start for note in all_notes])
    if len(starts) > 1:
        iois = np.diff(starts)
        features.append(np.mean(iois))
        features.append(np.std(iois))
    else:
        features.extend([0.0, 0.0])

    # Pitch intervals
    sorted_notes = sorted(all_notes, key=lambda x: x.start)
    if len(sorted_notes) > 1:
        intervals = [sorted_notes[i+1].pitch - sorted_notes[i].pitch
                     for i in range(len(sorted_notes)-1)]
        features.append(np.mean(np.abs(intervals)))
        features.append(np.std(intervals))
    else:
        features.extend([0.0, 0.0])

    return features[:20]


def extract_chroma_features(midi_data: pretty_midi.PrettyMIDI) -> List[float]:
    """Extract chroma (pitch class) features."""
    try:
        # Get chroma (12 pitch classes x time)
        chroma = midi_data.get_chroma(fs=config.MIDI_PIANO_ROLL_FS)

        if chroma.size == 0:
            return [0.0] * 12

        # Average chroma over time
        avg_chroma = np.mean(chroma, axis=1)

        # Normalize
        total = np.sum(avg_chroma)
        if total > 0:
            avg_chroma = avg_chroma / total

        return avg_chroma.tolist()

    except Exception:
        return [0.0] * 12


def normalize_features(features: np.ndarray) -> np.ndarray:
    """Normalize features to have zero mean and unit variance where appropriate."""
    # Simple min-max normalization to [0, 1]
    min_val = np.min(features)
    max_val = np.max(features)

    if max_val - min_val > 0:
        features = (features - min_val) / (max_val - min_val)

    return features


def get_temporal_midi_features(midi_path: str, num_frames: int = 50) -> np.ndarray:
    """
    Extract temporal MIDI features for sequence-level melody integration.

    Returns a sequence of feature vectors, one per time frame.
    This is useful for Approach 2 where we want to align melody with lyrics.
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
    except Exception:
        return np.zeros((num_frames, config.MIDI_FEATURE_DIM // 4))

    total_time = midi_data.get_end_time()
    if total_time == 0:
        return np.zeros((num_frames, config.MIDI_FEATURE_DIM // 4))

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

        if active_notes:
            pitches = [n.pitch for n in active_notes]
            velocities = [n.velocity for n in active_notes]

            features.append(np.mean(pitches) / 127)
            features.append(np.std(pitches) / 127 if len(pitches) > 1 else 0)
            features.append(np.mean(velocities) / 127)
            features.append(len(active_notes) / 20)  # Normalized note count
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])

        # Chroma for this frame
        try:
            piano_roll = midi_data.get_piano_roll(fs=config.MIDI_PIANO_ROLL_FS)
            start_idx = int(start_time * config.MIDI_PIANO_ROLL_FS)
            end_idx = int(end_time * config.MIDI_PIANO_ROLL_FS)

            if start_idx < piano_roll.shape[1]:
                end_idx = min(end_idx, piano_roll.shape[1])
                frame_roll = piano_roll[:, start_idx:end_idx]

                # Compute chroma
                chroma = np.zeros(12)
                for pitch in range(128):
                    chroma[pitch % 12] += np.sum(frame_roll[pitch, :])

                total = np.sum(chroma)
                if total > 0:
                    chroma = chroma / total
                features.extend(chroma.tolist())
            else:
                features.extend([0.0] * 12)
        except Exception:
            features.extend([0.0] * 12)

        # Tempo at this frame
        tempos = midi_data.get_tempo_changes()
        current_tempo = 120.0
        for t, tempo in zip(tempos[0], tempos[1]):
            if t <= start_time:
                current_tempo = tempo
        features.append(current_tempo / 200)  # Normalized tempo

        # Pad to fixed size per frame
        feature_dim = config.MIDI_FEATURE_DIM // 4
        while len(features) < feature_dim:
            features.append(0.0)
        features = features[:feature_dim]

        frame_features.append(features)

    return np.array(frame_features, dtype=np.float32)
